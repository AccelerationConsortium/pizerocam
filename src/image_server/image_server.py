"""
PiCamera unit HTTP server.

Endpoints (all responses are JSON unless noted):
    GET       /             Browser control page (live stream + light controls)
    GET       /status       Unit status: camera, light, stream
    GET|POST  /photo        Take a full-resolution photo -> image/jpeg
                              ?flash=true   turn the light on for the photo (then restore)
                              ?settle=SEC   wait before capturing (default: PhotoSettleSeconds
                                            when flash switched the light on, else 0)
    GET       /stream       Live MJPEG video (multipart/x-mixed-replace)
    GET       /light        Light state
    POST      /light/on     Turn light on; optional {"color": [r,g,b], "brightness": 0-1}
    POST      /light/off    Turn light off
    POST      /light/color  Set color/brightness without changing on/off state

POST parameters can be sent as a JSON body or as query parameters
(e.g. /light/on?color=255,0,0&brightness=0.5).

If a token is configured (--token or PIZEROCAM_TOKEN), every request must send
"Authorization: Bearer <token>" or "?token=<token>".
"""

import hmac
import json
import os
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import yaml

from logger import get_logger
from .network import detect_ip

__version__ = "2.0.0"

_SETTINGS_FILE = Path(__file__).resolve().parent / "server_settings.yaml"
_DEFAULT_SETTINGS = {
    "ServerPort": 2222,
    "PhotoResolution": "max",
    "PhotoSettleSeconds": 5,
    "StreamSize": [1280, 720],
    "LedCount": 12,
    "LightDefaultColor": [255, 255, 255],
    "LightBrightness": 1.0,
}
MAX_SETTLE_SECONDS = 30
MAX_BODY_BYTES = 4096


def load_settings(path=_SETTINGS_FILE):
    settings = dict(_DEFAULT_SETTINGS)
    try:
        with open(path, "r") as f:
            settings.update(yaml.safe_load(f) or {})
    except FileNotFoundError:
        pass
    return settings


class UnitError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


class CameraUnit:
    """Owns the camera and light hardware and implements the unit's operations."""

    def __init__(self, init_camera=True, init_light=True, resolution=None, settings=None,
                 camera=None, light=None, logger=None):
        self.settings = settings or load_settings()
        self.logger = logger or get_logger("PiCameraUnit")
        self.camera, self.camera_error = camera, None
        self.light, self.light_error = light, None
        self._photo_lock = threading.Lock()

        if self.light is None and init_light:
            try:
                from .hardware import Light
                self.light = Light(
                    num_pixels=self.settings["LedCount"],
                    color=self.settings["LightDefaultColor"],
                    brightness=self.settings["LightBrightness"],
                    logger=self.logger,
                )
                self.light.blink()
            except Exception as e:
                self.light_error = str(e)
                self.logger.error(f"Light initialization failed: {e}")

        if self.camera is None and init_camera:
            try:
                from .hardware import Camera
                self.camera = Camera(
                    resolution=resolution or self.settings["PhotoResolution"],
                    stream_size=self.settings["StreamSize"],
                    logger=self.logger,
                )
            except Exception as e:
                self.camera_error = str(e)
                self.logger.error(f"Camera initialization failed: {e}")

    def require_camera(self):
        if self.camera is None:
            raise UnitError(503, f"camera not available: {self.camera_error or 'disabled'}")
        return self.camera

    def require_light(self):
        if self.light is None:
            raise UnitError(503, f"light not available: {self.light_error or 'disabled'}")
        return self.light

    def status(self):
        return {
            "device": "picamera-unit",
            "version": __version__,
            "camera": self.camera.info() if self.camera else {"error": self.camera_error or "disabled"},
            "light": self.light.state() if self.light else {"error": self.light_error or "disabled"},
        }

    def take_photo(self, flash=False, settle=None):
        """Return (jpeg_bytes, filename)."""
        camera = self.require_camera()
        with self._photo_lock:
            switched_on = False
            if flash:
                light = self.require_light()
                if not light.is_on:
                    light.on()
                    switched_on = True
            if settle is None:
                settle = self.settings["PhotoSettleSeconds"] if switched_on else 0
            try:
                if settle:
                    time.sleep(settle)
                data = camera.capture_jpeg()
            finally:
                if switched_on:
                    self.light.off()
        filename = f"capture_{datetime.now().strftime('%Y%m%d-%H%M%S')}.jpg"
        self.logger.info(f"Captured {filename} ({len(data)} bytes, flash={flash})")
        return data, filename

    def close(self):
        for part in (self.camera, self.light):
            if part is not None:
                try:
                    part.close()
                except Exception as e:
                    self.logger.warning(f"Error closing {type(part).__name__}: {e}")


def _parse_bool(value, name):
    if value is None or isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off", ""):
        return False
    raise UnitError(400, f"{name} must be true or false")


def _parse_settle(value):
    if value is None:
        return None
    try:
        settle = float(value)
    except (TypeError, ValueError):
        raise UnitError(400, "settle must be a number of seconds")
    if not 0 <= settle <= MAX_SETTLE_SECONDS:
        raise UnitError(400, f"settle must be between 0 and {MAX_SETTLE_SECONDS} seconds")
    return settle


class _Handler(BaseHTTPRequestHandler):
    server_version = f"PiCameraUnit/{__version__}"

    # --- plumbing ---------------------------------------------------------

    @property
    def unit(self):
        return self.server.unit

    def log_message(self, fmt, *args):
        self.unit.logger.debug(f"{self.address_string()} - {fmt % args}")

    def _send(self, status, body, content_type, extra_headers=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status, payload):
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

    def _params(self, query):
        """Merge query parameters with a JSON body (body wins)."""
        params = {k: v[-1] for k, v in query.items()}
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            raise UnitError(413, "request body too large")
        if length:
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                raise UnitError(400, "body must be valid JSON")
            if not isinstance(body, dict):
                raise UnitError(400, "body must be a JSON object")
            params.update(body)
        return params

    def _authorized(self, query):
        token = self.server.token
        if not token:
            return True
        supplied = query.get("token", [""])[-1]
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            supplied = auth[len("Bearer "):].strip()
        return hmac.compare_digest(supplied.encode(), token.encode())

    def _dispatch(self, method):
        url = urlsplit(self.path)
        query = parse_qs(url.query)
        path = url.path.rstrip("/") or "/"
        route = self.ROUTES.get(path)
        try:
            if not self._authorized(query):
                raise UnitError(401, "missing or invalid token")
            if route is None:
                raise UnitError(404, f"unknown endpoint {path}")
            methods, handler_name = route
            if method not in methods:
                raise UnitError(405, f"{path} supports {', '.join(sorted(methods))}")
            getattr(self, handler_name)(self._params(query) if method == "POST" else
                                        {k: v[-1] for k, v in query.items()})
        except UnitError as e:
            self._json(e.status, {"error": e.message})
        except ValueError as e:
            self._json(400, {"error": str(e)})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            self.unit.logger.exception(f"Error handling {method} {path}")
            self._json(500, {"error": str(e)})

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    # --- endpoints --------------------------------------------------------

    def index(self, params):
        self._send(200, _INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")

    def status(self, params):
        self._json(200, self.unit.status())

    def photo(self, params):
        data, filename = self.unit.take_photo(
            flash=_parse_bool(params.get("flash"), "flash"),
            settle=_parse_settle(params.get("settle")),
        )
        self._send(200, data, "image/jpeg",
                   {"Content-Disposition": f'inline; filename="{filename}"'})

    def stream(self, params):
        camera = self.unit.require_camera()
        camera.open_stream()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seq = 0
            while not self.server.stopping.is_set():
                seq, frame = camera.wait_frame(seq, timeout=5.0)
                if frame is None:
                    continue
                self.wfile.write(b"--FRAME\r\nContent-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            camera.close_stream()

    def light_state(self, params):
        self._json(200, self.unit.require_light().state())

    def light_on(self, params):
        self._json(200, self.unit.require_light().on(params.get("color"), params.get("brightness")))

    def light_off(self, params):
        self._json(200, self.unit.require_light().off())

    def light_color(self, params):
        if params.get("color") is None and params.get("brightness") is None:
            raise UnitError(400, "provide color and/or brightness")
        self._json(200, self.unit.require_light().set(params.get("color"), params.get("brightness")))

    ROUTES = {
        "/": ({"GET"}, "index"),
        "/status": ({"GET"}, "status"),
        "/photo": ({"GET", "POST"}, "photo"),
        "/stream": ({"GET"}, "stream"),
        "/light": ({"GET"}, "light_state"),
        "/light/on": ({"POST"}, "light_on"),
        "/light/off": ({"POST"}, "light_off"),
        "/light/color": ({"POST"}, "light_color"),
    }


class _HTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, unit, token):
        self.unit = unit
        self.token = token
        self.stopping = threading.Event()
        super().__init__(address, _Handler)


class ImageServer:
    """
    PiCamera unit server: exposes the camera and LED light over HTTP.

    Example:
        server = ImageServer(port=2222)
        server.start(background=False)   # blocks; Ctrl+C to stop
    """

    def __init__(self, host="0.0.0.0", port=None, init_camera=True, init_light=True,
                 resolution=None, token=None, unit=None):
        settings = load_settings()
        self.host = host
        self.port = settings["ServerPort"] if port is None else port
        self.token = token if token is not None else os.getenv("PIZEROCAM_TOKEN") or None
        self.unit = unit or CameraUnit(init_camera, init_light, resolution, settings)
        self.logger = self.unit.logger
        self._ip = None
        self._httpd = None
        self._thread = None

    def get_ip_address(self):
        if self._ip is None:
            self._ip = detect_ip(self.logger)
        return self._ip

    def start(self, background=True):
        if self._httpd is not None:
            self.logger.warning("Server is already running")
            return False
        try:
            self._httpd = _HTTPServer((self.host, self.port), self.unit, self.token)
        except OSError as e:
            self.logger.error(f"Failed to start server on {self.host}:{self.port}: {e}")
            return False
        self.port = self._httpd.server_address[1]
        self.logger.info(f"PiCamera unit listening on http://{self.get_ip_address()}:{self.port}/"
                         f" (bound to {self.host}, auth {'on' if self.token else 'off'})")
        if background:
            self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
            self._thread.start()
        else:
            try:
                self._httpd.serve_forever()
            finally:
                self.stop()
        return True

    def stop(self, close_hardware=True):
        httpd, self._httpd = self._httpd, None
        if httpd is None:
            return False
        httpd.stopping.set()
        if self._thread is not None:
            httpd.shutdown()
            self._thread.join(timeout=5)
            self._thread = None
        httpd.server_close()
        if close_hardware:
            self.unit.close()
        self.logger.info("Server stopped")
        return True

    def is_running(self):
        return self._httpd is not None

    def get_logger(self):
        return self.logger

    def __enter__(self):
        self.start(background=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


_INDEX_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PiCamera Unit</title>
<style>
 body{font-family:system-ui,sans-serif;margin:0;padding:16px;background:#111;color:#eee}
 h1{font-size:1.2rem;margin:0 0 12px}
 img{max-width:100%;background:#000;border-radius:6px;display:block}
 .row{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:12px 0}
 button{padding:8px 14px;border:0;border-radius:6px;background:#2d6cdf;color:#fff;font-size:1rem}
 button.alt{background:#444}
 pre{background:#222;padding:8px;border-radius:6px;overflow:auto;font-size:.85rem}
</style></head><body>
<h1>PiCamera Unit</h1>
<img id="view" alt="camera view">
<div class="row">
 <button id="streamBtn">Start stream</button>
 <button id="photoBtn">Take photo</button>
 <label><input type="checkbox" id="flash"> flash</label>
</div>
<div class="row">
 <button id="onBtn">Light on</button>
 <button id="offBtn" class="alt">Light off</button>
 <input type="color" id="color" value="#ffffff">
 <input type="range" id="bright" min="0" max="1" step="0.05" value="1">
</div>
<pre id="out"></pre>
<script>
const token = new URLSearchParams(location.search).get("token");
const url = (p, q = {}) => { if (token) q.token = token;
  const s = new URLSearchParams(q).toString(); return p + (s ? "?" + s : ""); };
const out = document.getElementById("out"), view = document.getElementById("view");
const show = async r => { const t = await r.text(); out.textContent = r.status + " " + t; };
const post = (p, body) => fetch(url(p), {method: "POST",
  headers: {"Content-Type": "application/json"}, body: JSON.stringify(body || {})}).then(show);
const hex = h => [1, 3, 5].map(i => parseInt(h.substr(i, 2), 16));
let streaming = false, lastPhoto = null;
document.getElementById("streamBtn").onclick = e => {
  streaming = !streaming;
  view.src = streaming ? url("/stream") : "";
  e.target.textContent = streaming ? "Stop stream" : "Start stream";
};
document.getElementById("photoBtn").onclick = async () => {
  out.textContent = "capturing...";
  const r = await fetch(url("/photo", {flash: document.getElementById("flash").checked}));
  if (!r.ok) return show(r);
  if (lastPhoto) URL.revokeObjectURL(lastPhoto);
  lastPhoto = URL.createObjectURL(await r.blob());
  if (streaming) document.getElementById("streamBtn").click();
  view.src = lastPhoto; out.textContent = "photo captured (right-click to save)";
};
const lightBody = () => ({color: hex(document.getElementById("color").value),
  brightness: +document.getElementById("bright").value});
document.getElementById("onBtn").onclick = () => post("/light/on", lightBody());
document.getElementById("offBtn").onclick = () => post("/light/off");
document.getElementById("color").onchange = () => post("/light/color", lightBody());
document.getElementById("bright").onchange = () => post("/light/color", lightBody());
fetch(url("/status")).then(show);
</script></body></html>
"""

__all__ = ["ImageServer", "CameraUnit", "load_settings"]
