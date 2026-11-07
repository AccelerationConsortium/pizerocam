import json
import threading
import time
from pathlib import Path

import pytest
from sdl_lab_contract import EquipmentStatus, check_consistency

from conftest import JPEG, claim

FIXTURES = Path(__file__).parent / "fixtures"


def status(client):
    r = client.get("/status")
    assert r.status_code == 200
    env = EquipmentStatus.model_validate(r.json())
    assert check_consistency(env) == []
    return env


def test_probe_health_docs(make_client):
    c = make_client()
    assert c.get("/").json() == {"equipment_id": "lle_camera_unit", "equipment_name": "PiZeroCam Service",
                                 "protocol_version": "1.2"}
    assert c.get("/health").json() == {"status": "healthy"}
    assert c.get("/openapi.json").status_code == 200
    assert c.get("/docs").status_code == 200
    r = c.get("/agent-docs")
    assert r.headers["content-type"].startswith("text/markdown") and "pizerocam.snapshot" in r.text
    r = c.get("/llms.txt")
    assert r.headers["content-type"].startswith("text/plain") and "(agent-docs)" in r.text


def test_ready_status(make_client):
    env = status(make_client())
    assert (env.equipment_status, env.activity) == ("ready", "idle")
    assert env.allowed_actions == ["pizerocam.snapshot", "pizerocam.light.on", "pizerocam.light.off",
                                   "pizerocam.light.set"]
    assert env.components["camera"].state == "idle" and env.components["light"].state == "off"
    assert env.metrics["cycles_total"].value == 0
    assert env.details["claimed_by"] is None and env.details["light"]["on"] is False


@pytest.mark.parametrize("camera,light,state,allowed", [
    ("ok", "failed", "degraded", ["pizerocam.snapshot"]),
    ("ok", "disabled", "degraded", ["pizerocam.snapshot"]),
    ("failed", "ok", "error", ["pizerocam.light.on", "pizerocam.light.off", "pizerocam.light.set"]),
    ("disabled", "ok", "degraded", ["pizerocam.light.on", "pizerocam.light.off", "pizerocam.light.set"]),
    ("disabled", "disabled", "dry_run", []),
    ("failed", "failed", "error", []),
])
def test_health_reflects_subsystems(make_client, camera, light, state, allowed):
    c = make_client(camera=camera, light=light)
    env = status(c)
    assert env.equipment_status == state and env.allowed_actions == allowed
    assert env.message


def test_control_requires_claim(make_client):
    c = make_client()
    for path in ("/control/light/on", "/control/light/off", "/control/snapshot"):
        r = c.post(path, json={})
        assert r.status_code == 423 and r.json()["code"] == "claim_required"
    r = c.post("/control/light/on", json={}, headers={"X-Claim-Token": "bogus"})
    assert r.status_code == 423


def test_claim_lifecycle(make_client):
    c = make_client()
    h = claim(c)
    again = claim(c)  # same owner+session: idempotent
    assert again == h
    env = status(c)
    assert env.details["claimed_by"]["owner"] == "tester"
    r = c.post("/control/claim", json={"owner": "other", "session_id": "s2"})
    assert r.status_code == 409 and r.json()["claimed_by"]["owner"] == "tester"
    assert int(r.headers["Retry-After"]) >= 1
    assert c.post("/control/heartbeat", headers=h).status_code == 200
    assert c.post("/control/heartbeat", headers={"X-Claim-Token": "bogus"}).status_code == 401
    assert c.post("/control/release", headers=h).status_code == 204
    assert c.post("/control/release", headers=h).status_code == 204  # idempotent
    assert status(c).details["claimed_by"] is None


def test_claim_ttl_is_clamped_and_expires(make_client):
    c = make_client()
    r = c.post("/control/claim", json={"owner": "o", "session_id": "s", "ttl_s": 0.01})
    assert r.json()["heartbeat_interval_s"] == pytest.approx(5 / 3)
    c.service.claim["deadline"] = c.service.clock() - 1  # simulate expiry
    assert status(c).details["claimed_by"] is None
    r = c.post("/control/light/on", json={}, headers={"X-Claim-Token": r.json()["claim_token"]})
    assert r.status_code == 423


def test_light_controls(make_client):
    c = make_client()
    h = claim(c)
    r = c.post("/control/light/on", json={"color": [255, 0, 0], "brightness": 0.5}, headers=h)
    assert r.json() == {"on": True, "color": [255, 0, 0], "brightness": 0.5}
    assert c.pixels.last == (255, 0, 0) and c.pixels.brightness == 0.5
    r = c.post("/control/light/set", json={"color": [0, 255, 0]}, headers=h)
    assert r.json()["on"] is True and c.pixels.last == (0, 255, 0)
    r = c.post("/control/light/off", headers=h)
    assert r.json()["on"] is False and c.pixels.last == (0, 0, 0)
    r = c.post("/control/light/set", json={"brightness": 0.2}, headers=h)
    assert r.json() == {"on": False, "color": [0, 255, 0], "brightness": 0.2} and c.pixels.last == (0, 0, 0)
    env = status(c)
    assert env.components["light"].state == "off" and "rgb(0,255,0) at 20%" == env.components["light"].message
    # Releasing the claim leaves the light alone.
    c.post("/control/light/on", json={}, headers=h)
    c.post("/control/release", headers=h)
    assert c.pixels.last == (0, 255, 0)


@pytest.mark.parametrize("path,body", [
    ("/control/light/on", {"color": [256, 0, 0]}),
    ("/control/light/on", {"color": [1, 2]}),
    ("/control/light/on", {"brightness": 1.5}),
    ("/control/light/on", {"colour": [1, 2, 3]}),
    ("/control/light/set", {}),
    ("/control/snapshot", {"settle_s": 9}),
    ("/control/snapshot", {"flash": "maybe"}),
])
def test_validation(make_client, path, body):
    c = make_client()
    r = c.post(path, json=body, headers=claim(c))
    assert r.status_code == 422 and r.json()["code"] == "invalid_request"


def test_unavailable_parts_refuse(make_client):
    c = make_client(camera="failed", light="disabled")
    h = claim(c)
    r = c.post("/control/snapshot", json={}, headers=h)
    assert r.status_code == 409 and r.json()["code"] == "camera_unavailable"
    r = c.post("/control/light/on", json={}, headers=h)
    assert r.status_code == 409 and r.json()["code"] == "light_unavailable"
    assert c.get("/stream").status_code == 409
    assert status(c).last_error is None  # refusals never set last_error


def test_snapshot_and_media(make_client):
    c = make_client()
    h = claim(c)
    r = c.post("/control/snapshot", json={}, headers=h)
    assert r.status_code == 200, r.text
    snap = r.json()
    assert snap["url"].startswith("/media/snapshots/main/capture_") and snap["bytes"] == len(JPEG)
    assert (snap["width"], snap["height"], snap["lens"]) == (4608, 2592, "main")
    img = c.get(snap["url"])
    assert img.status_code == 200 and img.content == JPEG and img.headers["content-type"] == "image/jpeg"
    listing = c.get("/media").json()
    assert listing["snapshots"][0]["url"] == snap["url"] and listing["recordings"] == []
    env = status(c)
    assert env.metrics["cycles_total"].value == 1 and env.details["last_snapshot"]["url"] == snap["url"]
    assert c.pixels.last == (0, 0, 0)  # no flash
    # an empty body is accepted too
    assert c.post("/control/snapshot", headers=h).status_code == 200


def test_media_is_capped_and_paths_are_confined(make_client):
    c = make_client()
    h = claim(c)
    urls = [c.post("/control/snapshot", json={}, headers=h).json()["url"] for _ in range(5)]
    listing = c.get("/media").json()["snapshots"]
    assert [m["url"] for m in listing] == list(reversed(urls[-3:]))
    assert c.get(urls[0]).status_code == 404
    for bad in ("/media/snapshots/main/..%2F..%2Fetc%2Fpasswd", "/media/recordings/main/x.jpg",
                "/media/snapshots/main/server_settings.yaml"):
        assert c.get(bad).status_code == 404


def test_flash_snapshot_restores_light(make_client):
    c = make_client(settle=0.1)
    h = claim(c)
    c.post("/control/light/set", json={"color": [0, 0, 255]}, headers=h)
    seen = []
    orig = c.camera.capture_jpeg
    c.camera.capture_jpeg = lambda: (seen.append(c.pixels.last), orig())[1]
    t = time.monotonic()
    assert c.post("/control/snapshot", json={"flash": True}, headers=h).status_code == 200
    assert time.monotonic() - t >= 0.1
    assert seen == [(0, 0, 255)] and c.pixels.last == (0, 0, 0)
    # Light already on: left on, no settle by default.
    c.post("/control/light/on", json={}, headers=h)
    c.post("/control/snapshot", json={"flash": True}, headers=h)
    assert c.pixels.last == (0, 0, 255)


def test_flash_needs_light(make_client):
    c = make_client(light="failed")
    r = c.post("/control/snapshot", json={"flash": True}, headers=claim(c))
    assert r.status_code == 409 and r.json()["code"] == "light_unavailable"
    assert c.post("/control/snapshot", json={}, headers=claim(c)).status_code == 200


def test_busy_while_capturing(make_client):
    c = make_client()
    h = claim(c)
    c.camera.gate = threading.Event()
    result = {}
    t = threading.Thread(target=lambda: result.update(r=c.post("/control/snapshot", json={}, headers=h)))
    t.start()
    deadline = time.monotonic() + 3
    while not c.service.capturing and time.monotonic() < deadline:
        time.sleep(0.01)
    env = status(c)
    assert (env.equipment_status, env.activity) == ("busy", "running")
    assert "pizerocam.snapshot" not in env.allowed_actions
    assert env.components["camera"].state == "capturing"
    r = c.post("/control/snapshot", json={}, headers=h)
    assert r.status_code == 409 and r.json()["code"] == "capture_in_progress"
    c.camera.gate.set()
    t.join(5)
    assert result["r"].status_code == 200
    env = status(c)
    assert (env.equipment_status, env.activity) == ("ready", "idle") and env.activity_since


def test_capture_failure_sets_and_clears_last_error(make_client):
    c = make_client()
    h = claim(c)
    c.camera.fail = True
    r = c.post("/control/snapshot", json={}, headers=h)
    assert r.status_code == 500 and r.json()["code"] == "capture_failed"
    env = status(c)
    assert env.last_error.code == "capture_failed" and env.equipment_status == "ready"
    c.camera.fail = False
    assert c.post("/control/snapshot", json={}, headers=h).status_code == 200
    assert status(c).last_error is None


def test_light_write_failure(make_client):
    c = make_client()
    h = claim(c)
    c.pixels.fail = True
    r = c.post("/control/light/on", json={}, headers=h)
    assert r.status_code == 500 and r.json()["code"] == "light_write_failed"
    assert status(c).last_error.code == "light_write_failed"
    c.pixels.fail = False
    assert c.post("/control/light/off", headers=h).status_code == 200
    assert status(c).last_error is None


def test_stream_counts_viewers(make_client):
    """Real uvicorn server: TestClient never delivers a client disconnect to a stream."""
    import socket

    import httpx
    import uvicorn

    c = make_client()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(c.app, host="127.0.0.1", port=port, log_level="warning",
                                           lifespan="off"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        while not server.started:
            time.sleep(0.02)
        with httpx.stream("GET", f"http://127.0.0.1:{port}/stream", timeout=5) as r:
            assert r.headers["content-type"].startswith("multipart/x-mixed-replace")
            buf = b""
            for chunk in r.iter_bytes():
                buf += chunk
                if buf.count(b"--FRAME") >= 3:
                    break
            assert c.camera.viewers == 1
            assert status(c).components["camera"].state == "streaming"
        deadline = time.monotonic() + 5
        while c.camera.viewers and time.monotonic() < deadline:
            time.sleep(0.05)
        assert c.camera.viewers == 0
        assert status(c).components["camera"].state == "idle"
    finally:
        server.should_exit = True
        thread.join(5)


@pytest.mark.parametrize("path", sorted(FIXTURES.glob("status_*.json")), ids=lambda p: p.stem)
def test_fixtures_are_spec_consistent(path):
    env = EquipmentStatus.model_validate(json.loads(path.read_text()))
    assert env.protocol_version == "1.2"
    assert check_consistency(env) == []
