"""Hardware-wrapper checks that need no Pi (picamera2 is not imported)."""

import io
import logging
import threading

import pytest

from image_server.hardware import Camera, _FrameBuffer


def test_frame_buffer_is_accepted_by_picamera2_file_output():
    # picamera2's FileOutput raises "Must pass io.BufferedIOBase" for anything else.
    buf = _FrameBuffer()
    assert isinstance(buf, io.BufferedIOBase)
    assert buf.writable()


def test_frame_buffer_stays_usable_after_the_encoder_closes_it():
    buf = _FrameBuffer()
    assert buf.write(b"frame-1") == 7
    buf.flush()
    buf.close()  # FileOutput closes its target when the encoder stops
    buf.write(memoryview(b"frame-2"))
    assert buf.wait_frame(1, timeout=0.1) == (2, b"frame-2")


def _bare_camera(start_encoder):
    cam = Camera.__new__(Camera)
    cam.logger = logging.getLogger("test")
    cam._lock = threading.Lock()
    cam._frames = _FrameBuffer()
    cam._encoder_running = False
    cam._viewers = 0
    cam._start_encoder = start_encoder
    return cam


def test_failed_encoder_start_does_not_leak_a_viewer():
    def boom():
        raise RuntimeError("encoder failed")

    cam = _bare_camera(boom)
    with pytest.raises(RuntimeError):
        cam.open_stream()
    assert cam._viewers == 0


def test_viewer_counted_once_the_encoder_runs():
    cam = _bare_camera(lambda: setattr(cam, "_encoder_running", True))
    cam.open_stream()
    cam.open_stream()
    assert cam._viewers == 2
