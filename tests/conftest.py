import threading
import time

import pytest
from fastapi.testclient import TestClient

from image_server.app import create_app
from image_server.hardware import Light, _FrameBuffer
from image_server.media import MediaStore
from image_server.service import DEFAULT_SETTINGS, CameraUnitService, Subsystem

JPEG = b"\xff\xd8fake-jpeg\xff\xd9"


class FakePixels:
    def __init__(self):
        self.brightness = 1.0
        self.last = (0, 0, 0)
        self.fail = False

    def fill(self, color):
        if self.fail:
            raise OSError("spi write failed")
        self.last = tuple(color)


class FakeCamera:
    still_size = (4608, 2592)

    def __init__(self):
        self.frames = _FrameBuffer()
        self.viewers = 0
        self.captures = 0
        self.fail = False
        self.gate = None  # threading.Event: block capture until set
        self.during_capture = []
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        i = 0
        while not self._stop.is_set():
            if self.viewers:
                self.frames.write(b"\xff\xd8frame%d\xff\xd9" % i)
                i += 1
            time.sleep(0.01)

    def capture_jpeg(self):
        if self.gate is not None:
            self.gate.wait(5)
        if self.fail:
            raise RuntimeError("sensor timeout")
        self.captures += 1
        return JPEG

    def open_stream(self):
        self.viewers += 1

    def close_stream(self):
        self.viewers -= 1

    def wait_frame(self, seq, timeout=5.0):
        return self.frames.wait_frame(seq, timeout)

    def info(self):
        return {"model": "fake", "still_size": list(self.still_size), "stream_size": [1280, 720],
                "streaming": self.viewers > 0, "stream_viewers": self.viewers}

    def close(self):
        self._stop.set()


@pytest.fixture
def make_client(tmp_path):
    clients = []

    def make(camera="ok", light="ok", settle=0.05):
        settings = dict(DEFAULT_SETTINGS, PhotoSettleSeconds=settle, MediaDir=str(tmp_path / "media"))
        parts = {}
        pixels = FakePixels()
        cam = FakeCamera()
        parts["camera"] = {"ok": Subsystem(cam), "failed": Subsystem(error="no camera detected"),
                           "disabled": Subsystem(disabled=True)}[camera]
        parts["light"] = {"ok": Subsystem(Light(pixels=pixels)), "failed": Subsystem(error="SPI not enabled"),
                          "disabled": Subsystem(disabled=True)}[light]
        service = CameraUnitService(settings, parts["camera"], parts["light"],
                                    MediaStore(settings["MediaDir"], 3))
        client = TestClient(create_app(lambda: service))
        client.__enter__()
        clients.append(client)
        client.service, client.pixels, client.camera = service, pixels, cam
        return client

    yield make
    for c in clients:
        c.__exit__(None, None, None)


def claim(client, owner="tester", session="s1", ttl=30):
    r = client.post("/control/claim", json={"owner": owner, "session_id": session, "ttl_s": ttl})
    assert r.status_code == 200, r.text
    return {"X-Claim-Token": r.json()["claim_token"]}
