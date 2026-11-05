"""
Hardware wrappers for the PiCamera unit: the Pi camera (stills + MJPEG stream)
and the NeoPixel LED ring. Hardware libraries are imported lazily so the rest of
the package can be imported (and tested) on machines without them.
"""

import io
import logging
import threading
import time


class _FrameBuffer:
    """File-like sink for the MJPEG encoder; hands the latest frame to stream viewers."""

    def __init__(self):
        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0

    def write(self, buf):
        with self._cond:
            self._frame = bytes(buf)
            self._seq += 1
            self._cond.notify_all()
        return len(buf)

    def flush(self):
        pass

    def close(self):
        # FileOutput closes its target when the encoder stops; keep the buffer reusable.
        pass

    def wait_frame(self, last_seq, timeout):
        """Block until a frame newer than last_seq arrives. Returns (seq, frame) or (last_seq, None)."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq > last_seq, timeout):
                return last_seq, None
            return self._seq, self._frame


class Camera:
    """
    Pi camera running continuously in a video configuration (for streaming).
    Stills are taken by briefly switching to a full-resolution still configuration.
    """

    RESOLUTIONS = {
        "max": None,  # sensor's full resolution
        "4k": (3840, 2160),
        "fhd": (1920, 1080),
        "hd": (1280, 720),
        "vga": (640, 480),
    }

    def __init__(self, resolution="max", stream_size=(1280, 720), logger=None):
        from picamera2 import Picamera2

        self.logger = logger or logging.getLogger(__name__)
        self._cam = Picamera2(0)
        self._lock = threading.Lock()
        self._frames = _FrameBuffer()
        self._encoder_running = False
        self._viewers = 0

        controls = self._autofocus_controls()
        self._still_config = self._pick_still_config(resolution, controls)
        self._video_config = self._cam.create_video_configuration(
            main={"size": tuple(stream_size)}, controls=controls
        )
        self._cam.configure(self._video_config)
        self._cam.start()
        self.logger.info(f"Camera ready: stills {self.still_size}, stream {self.stream_size}")

    def _autofocus_controls(self):
        if "AfMode" in self._cam.camera_controls:
            from libcamera import controls
            return {"AfMode": controls.AfModeEnum.Continuous}
        return {}

    def _pick_still_config(self, resolution, controls):
        sensor = tuple(getattr(self._cam, "sensor_resolution", (4608, 2592)))
        candidates = []
        if resolution in self.RESOLUTIONS:
            candidates.append(self.RESOLUTIONS[resolution] or sensor)
        candidates += [sensor, (1920, 1080), (1280, 720), (640, 480)]

        tried = []
        for size in candidates:
            if size in tried:
                continue
            tried.append(size)
            try:
                config = self._cam.create_still_configuration(main={"size": size}, controls=controls)
                self._cam.configure(config)  # validate that buffers can be allocated
                return config
            except Exception as e:
                self.logger.warning(f"Still resolution {size[0]}x{size[1]} unavailable: {e}")
        raise RuntimeError("Could not configure the camera at any still resolution")

    @property
    def still_size(self):
        return tuple(self._still_config["main"]["size"])

    @property
    def stream_size(self):
        return tuple(self._video_config["main"]["size"])

    @property
    def streaming(self):
        return self._encoder_running

    def info(self):
        return {
            "model": self._cam.camera_properties.get("Model"),
            "still_size": list(self.still_size),
            "stream_size": list(self.stream_size),
            "streaming": self._encoder_running,
            "stream_viewers": self._viewers,
        }

    # --- stills -----------------------------------------------------------

    def capture_jpeg(self):
        """Capture a full-resolution JPEG and return its bytes. Pauses the stream briefly."""
        with self._lock:
            was_streaming = self._encoder_running
            if was_streaming:
                self._stop_encoder()
            try:
                buf = io.BytesIO()
                self._cam.switch_mode_and_capture_file(self._still_config, buf, format="jpeg")
                return buf.getvalue()
            finally:
                if was_streaming:
                    self._start_encoder()

    # --- streaming --------------------------------------------------------

    def _start_encoder(self):
        from picamera2.outputs import FileOutput
        try:
            from picamera2.encoders import MJPEGEncoder  # hardware encoder (Pi Zero 2 / 4)
            self._cam.start_encoder(MJPEGEncoder(), FileOutput(self._frames))
        except Exception as e:
            self.logger.warning(f"Hardware MJPEG encoder unavailable ({e}); using software JPEG")
            from picamera2.encoders import JpegEncoder
            self._cam.start_encoder(JpegEncoder(), FileOutput(self._frames))
        self._encoder_running = True

    def _stop_encoder(self):
        self._cam.stop_encoder()
        self._encoder_running = False

    def open_stream(self):
        """Register a stream viewer; starts the encoder for the first one."""
        with self._lock:
            self._viewers += 1
            if not self._encoder_running:
                self._start_encoder()
                self.logger.info("Video stream started")

    def close_stream(self):
        """Unregister a stream viewer; stops the encoder after the last one leaves."""
        with self._lock:
            self._viewers = max(0, self._viewers - 1)
            if self._viewers == 0 and self._encoder_running:
                self._stop_encoder()
                self.logger.info("Video stream stopped (no viewers)")

    def wait_frame(self, last_seq, timeout=5.0):
        return self._frames.wait_frame(last_seq, timeout)

    def close(self):
        with self._lock:
            if self._encoder_running:
                self._stop_encoder()
            self._cam.stop()
            self._cam.close()
        self.logger.info("Camera closed")


class Light:
    """NeoPixel LED ring with on/off, color and brightness state."""

    def __init__(self, num_pixels=12, color=(255, 255, 255), brightness=1.0, pixels=None, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        if pixels is None:
            import board
            import neopixel
            # NeoPixel ring MUST use board.D10 (SPI MOSI) on the Pi
            pixels = neopixel.NeoPixel(board.D10, num_pixels, brightness=brightness, auto_write=True)
        self._pixels = pixels
        self._lock = threading.Lock()
        self.color = validate_color(color)
        self.brightness = validate_brightness(brightness)
        self.is_on = False
        self._pixels.fill((0, 0, 0))

    def _apply(self):
        self._pixels.brightness = self.brightness
        self._pixels.fill(self.color if self.is_on else (0, 0, 0))

    def _update(self, color, brightness):
        if color is not None:
            self.color = validate_color(color)
        if brightness is not None:
            self.brightness = validate_brightness(brightness)

    def on(self, color=None, brightness=None):
        with self._lock:
            self._update(color, brightness)
            self.is_on = True
            self._apply()
            return self._state()

    def off(self):
        with self._lock:
            self.is_on = False
            self._apply()
            return self._state()

    def set(self, color=None, brightness=None):
        """Change color/brightness. Applied immediately if on, otherwise used at next on()."""
        with self._lock:
            self._update(color, brightness)
            self._apply()
            return self._state()

    def blink(self, times=3, interval=0.3):
        with self._lock:
            for _ in range(times):
                self._pixels.fill((100, 100, 100))
                time.sleep(interval)
                self._pixels.fill((0, 0, 0))
                time.sleep(interval)
            self._apply()

    def _state(self):
        return {"on": self.is_on, "color": list(self.color), "brightness": self.brightness}

    def state(self):
        with self._lock:
            return self._state()

    def close(self):
        self.off()


def validate_color(value):
    """Accept [r,g,b], (r,g,b), {"r":..,"g":..,"b":..}, "r,g,b" or "#rrggbb"; return an (r, g, b) tuple."""
    if isinstance(value, dict):
        value = [value.get(k) for k in ("r", "g", "b")]
    elif isinstance(value, str):
        text = value.strip()
        if text.startswith("#") and len(text) == 7:
            value = [int(text[i:i + 2], 16) for i in (1, 3, 5)]
        else:
            value = text.split(",")
    try:
        r, g, b = (int(v) for v in value)
    except (TypeError, ValueError):
        raise ValueError("color must be [r, g, b], 'r,g,b' or '#rrggbb'")
    if not all(0 <= v <= 255 for v in (r, g, b)):
        raise ValueError("color values must be between 0 and 255")
    return (r, g, b)


def validate_brightness(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError("brightness must be a number between 0 and 1")
    if not 0.0 <= value <= 1.0:
        raise ValueError("brightness must be between 0 and 1")
    return value
