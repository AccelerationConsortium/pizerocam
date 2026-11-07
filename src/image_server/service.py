"""
Device state for the PiCamera unit: hardware, claims, activity and the
STATUS_SPEC v1.2 envelope. The FastAPI layer in ``app.py`` is a thin
presentation over this class.

Primary operation (STATUS_SPEC §2.3): acquiring a still photo. ``activity`` is
``running`` from the moment a snapshot is accepted until the JPEG is saved.
Streaming video and the light are reported in ``components`` / ``details``;
neither counts as the primary operation.
"""

import logging
import secrets
import socket
import threading
import time
from datetime import datetime, timedelta, timezone

import yaml
from sdl_lab_contract import (
    ClaimedBy,
    ClaimResponse,
    ComponentStatus,
    EquipmentStatus,
    ErrorInfo,
    MetricValue,
    ProbeResponse,
)

from . import __version__
from .models import (
    SKILL_LIGHT_OFF,
    SKILL_LIGHT_ON,
    SKILL_LIGHT_SET,
    SKILL_SNAPSHOT,
    LightState,
    SnapshotResponse,
)

PROTOCOL_VERSION = "1.2"
CLAIM_TTL_MIN_S = 5.0
CLAIM_TTL_MAX_S = 300.0
# last_error.code taxonomy (documented in README / agent guide)
ERROR_CODES = frozenset({"capture_failed", "light_write_failed"})

LOG = logging.getLogger("pizerocam")

DEFAULT_SETTINGS = {
    "EquipmentId": "lle_camera_unit",
    "EquipmentName": "PiZeroCam Service",
    "ServerPort": 2222,
    "PhotoResolution": "max",
    "PhotoSettleSeconds": 5,
    "StreamSize": [1280, 720],
    "MediaDir": "~/pizerocam-media",
    "MediaMaxFiles": 200,
    "LedCount": 12,
    "LightDefaultColor": [255, 255, 255],
    "LightBrightness": 1.0,
}


def load_settings(path=None):
    from pathlib import Path
    path = Path(path) if path else Path(__file__).resolve().parent / "server_settings.yaml"
    settings = dict(DEFAULT_SETTINGS)
    try:
        with open(path) as f:
            settings.update(yaml.safe_load(f) or {})
    except FileNotFoundError:
        pass
    return settings


def utcnow():
    return datetime.now(timezone.utc)


class ControlError(Exception):
    """A refusal or failure with an HTTP status and a stable ``code``."""

    def __init__(self, status, code, detail, **extra):
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail
        self.extra = extra

    def body(self):
        return {"detail": self.detail, "code": self.code, **self.extra}


class Subsystem:
    """Init outcome for one hardware part: the object, or why it is missing."""

    def __init__(self, device=None, error=None, disabled=False):
        self.device = device
        self.error = error
        self.disabled = disabled

    @property
    def ok(self):
        return self.device is not None


class CameraUnitService:
    def __init__(self, settings, camera: Subsystem, light: Subsystem, media, *, clock=time.monotonic):
        self.settings = settings
        self.camera = camera
        self.light = light
        self.media = media
        self.clock = clock
        self.started = clock()
        self.lock = threading.RLock()
        self.claim = None
        self.last_error = None
        self.capturing = False
        self.activity_since = utcnow()
        self.cycles_total = 0
        self.last_snapshot = None
        self.closed = False

    # --- identity ------------------------------------------------------------

    @property
    def equipment_id(self):
        return self.settings["EquipmentId"]

    @property
    def equipment_name(self):
        return self.settings["EquipmentName"]

    def probe(self):
        return ProbeResponse(equipment_id=self.equipment_id, equipment_name=self.equipment_name,
                             protocol_version=PROTOCOL_VERSION)

    # --- claims (STATUS_SPEC §5, hard enforcement) ---------------------------
    # Claims only serialize control. Expiry or release changes no hardware: the
    # light keeps its state and an in-flight capture finishes.

    def _expire(self):
        if self.claim and self.clock() >= self.claim["deadline"]:
            LOG.info("claim by %s expired", self.claim["owner"])
            self.claim = None

    def _claimed_by(self):
        if not self.claim:
            return None
        remaining = max(0.0, self.claim["deadline"] - self.clock())
        return ClaimedBy(session_id=self.claim["session_id"], owner=self.claim["owner"],
                         expires_at=utcnow() + timedelta(seconds=remaining))

    def _claim_response(self):
        claimed = self._claimed_by()
        return ClaimResponse(claim_token=self.claim["token"],
                             heartbeat_interval_s=self.claim["ttl"] / 3,
                             expires_at=claimed.expires_at)

    def acquire(self, request):
        with self.lock:
            self._expire()
            if self.closed:
                raise ControlError(503, "shutting_down", "service is shutting down")
            if self.claim:
                if self.claim["session_id"] == request.session_id and self.claim["owner"] == request.owner:
                    return self._claim_response()  # idempotent; heartbeats extend the lease
                remaining = max(0.0, self.claim["deadline"] - self.clock())
                raise ControlError(409, "claimed", "device is claimed by another session",
                                   claimed_by=self._claimed_by().model_dump(mode="json"),
                                   retry_after_s=round(remaining, 1))
            ttl = min(max(float(request.ttl_s), CLAIM_TTL_MIN_S), CLAIM_TTL_MAX_S)
            self.claim = {"token": secrets.token_urlsafe(32), "owner": request.owner,
                          "session_id": request.session_id, "ttl": ttl,
                          "deadline": self.clock() + ttl}
            LOG.info("claim acquired by %s", request.owner)
            return self._claim_response()

    def _token_matches(self, token):
        return bool(self.claim and token and secrets.compare_digest(token, self.claim["token"]))

    def _require_claim(self, token):
        self._expire()
        if self.closed:
            raise ControlError(503, "shutting_down", "service is shutting down")
        if not self._token_matches(token):
            claimed = self._claimed_by()
            raise ControlError(423, "claim_required", "a live X-Claim-Token is required",
                               claimed_by=claimed.model_dump(mode="json") if claimed else None)

    def heartbeat(self, token):
        with self.lock:
            self._expire()
            if not self._token_matches(token):
                raise ControlError(401, "claim_lost", "claim token is unknown or expired")
            self.claim["deadline"] = self.clock() + self.claim["ttl"]
            return self._claim_response()

    def release(self, token):
        with self.lock:
            self._expire()
            if self._token_matches(token):
                LOG.info("claim released by %s", self.claim["owner"])
                self.claim = None
            # Idempotent: unknown or already-released tokens also succeed (§5).

    # --- errors ----------------------------------------------------------------

    def _fault(self, code, message):
        assert code in ERROR_CODES, code
        self.last_error = ErrorInfo(code=code, message=message, severity="error", timestamp=utcnow())
        LOG.error("%s: %s", code, message)

    def _succeeded(self):
        self.last_error = None  # §6.4: clear on the next successful operational action

    # --- light -------------------------------------------------------------------

    def _require_light(self):
        if not self.light.ok:
            reason = "disabled" if self.light.disabled else f"failed to initialize: {self.light.error}"
            raise ControlError(409, "light_unavailable", f"light {reason}")
        return self.light.device

    def _light_call(self, fn):
        try:
            state = fn()
        except Exception as exc:
            self._fault("light_write_failed", f"light write failed: {exc}")
            raise ControlError(500, "light_write_failed", "light write failed; light state unknown") from exc
        self._succeeded()
        return LightState(**state)

    def light_on(self, body, token):
        with self.lock:
            self._require_claim(token)
            light = self._require_light()
            return self._light_call(lambda: light.on(body.color, body.brightness))

    def light_off(self, token):
        with self.lock:
            self._require_claim(token)
            light = self._require_light()
            return self._light_call(light.off)

    def light_set(self, body, token):
        with self.lock:
            self._require_claim(token)
            light = self._require_light()
            return self._light_call(lambda: light.set(body.color, body.brightness))

    # --- snapshot (primary operation) ------------------------------------------

    def _require_camera(self):
        if not self.camera.ok:
            reason = "disabled" if self.camera.disabled else f"failed to initialize: {self.camera.error}"
            raise ControlError(409, "camera_unavailable", f"camera {reason}")
        return self.camera.device

    def _set_capturing(self, value):
        if self.capturing != value:
            self.capturing = value
            self.activity_since = utcnow()

    def snapshot(self, body, token):
        with self.lock:
            self._require_claim(token)
            camera = self._require_camera()
            if self.capturing:
                raise ControlError(409, "capture_in_progress", "a snapshot is already being captured")
            light = self._require_light() if body.flash else None
            self._set_capturing(True)
        # The capture runs outside the lock so /status stays responsive.
        switched_on = False
        flash_off_failed = False
        try:
            if light is not None and not light.is_on:
                light.on()
                switched_on = True
            settle = body.settle_s
            if settle is None:
                settle = float(self.settings["PhotoSettleSeconds"]) if switched_on else 0.0
            if settle:
                time.sleep(settle)
            data = camera.capture_jpeg()
            taken_at = utcnow()
            name = self.media.save(data, taken_at)
        except Exception as exc:
            with self.lock:
                self._fault("capture_failed", f"snapshot failed: {exc}")
                self._set_capturing(False)
            raise ControlError(500, "capture_failed", f"snapshot failed: {exc}") from exc
        finally:
            if switched_on:
                try:
                    light.off()
                except Exception as exc:
                    flash_off_failed = True
                    with self.lock:
                        self._fault("light_write_failed", f"flash could not be switched off: {exc}")
        with self.lock:
            self._set_capturing(False)
            self.cycles_total += 1
            if not flash_off_failed:
                self._succeeded()
            width, height = camera.still_size
            self.last_snapshot = SnapshotResponse(
                path=self.media.rel_path(name), url=self.media.url_for(name),
                taken_at=taken_at.isoformat(), lens="main",
                width=width, height=height, bytes=len(data),
            )
            LOG.info("snapshot %s (%d bytes, flash=%s)", name, len(data), body.flash)
            return self.last_snapshot

    # --- status (side-effect-free apart from claim expiry bookkeeping) ----------

    def _camera_component(self):
        if not self.camera.ok:
            state = "disabled" if self.camera.disabled else "failed"
            return ComponentStatus(connected=False, state=state, message=self.camera.error)
        info = self.camera.device.info()
        state = "capturing" if self.capturing else ("streaming" if info.get("streaming") else "idle")
        return ComponentStatus(connected=True, state=state)

    def _light_component(self):
        if not self.light.ok:
            state = "disabled" if self.light.disabled else "failed"
            return ComponentStatus(connected=False, state=state, message=self.light.error)
        s = self.light.device.state()
        r, g, b = s["color"]
        return ComponentStatus(connected=True, state="on" if s["on"] else "off",
                               message=f"rgb({r},{g},{b}) at {round(s['brightness'] * 100)}%")

    def _health(self):
        """(equipment_status, message) per STATUS_SPEC §2.2."""
        cam, light = self.camera, self.light
        if not cam.ok and not light.ok and cam.disabled and light.disabled:
            return "dry_run", "camera and light disabled (no hardware)"
        if not cam.ok:
            if cam.disabled:
                return "degraded", "camera disabled; light only"
            return "error", f"camera failed to initialize: {cam.error}"
        if not light.ok:
            why = "disabled" if light.disabled else f"failed to initialize: {light.error}"
            return "degraded", f"light {why}; photos without flash still available"
        return ("busy" if self.capturing else "ready"), None

    def allowed_actions(self):
        actions = []
        if self.camera.ok and not self.capturing:
            actions.append(SKILL_SNAPSHOT)
        if self.light.ok:
            actions += [SKILL_LIGHT_ON, SKILL_LIGHT_OFF, SKILL_LIGHT_SET]
        return actions

    def status(self):
        with self.lock:
            self._expire()
            state, message = self._health()
            if state == "busy":
                message = "capturing a snapshot"
            details = {
                "claimed_by": self._claimed_by().model_dump(mode="json") if self.claim else None,
                "monitoring_only": False,
                "light": self.light.device.state() if self.light.ok else None,
                "camera": self.camera.device.info() if self.camera.ok else None,
                "last_snapshot": self.last_snapshot.model_dump() if self.last_snapshot else None,
                "media_count": self.media.count(),
                "stream_path": "/stream",
            }
            metrics = {"cycles_total": MetricValue(value=self.cycles_total, unit="count")}
            if self.camera.ok:
                metrics["stream_viewers"] = MetricValue(
                    value=self.camera.device.info().get("stream_viewers", 0), unit="count")
            return EquipmentStatus(
                protocol_version=PROTOCOL_VERSION,
                equipment_id=self.equipment_id,
                equipment_name=self.equipment_name,
                equipment_kind="other",
                equipment_version=__version__,
                host=socket.gethostname(),
                equipment_status=state,
                message=message,
                allowed_actions=self.allowed_actions(),
                activity="running" if self.capturing else "idle",
                activity_since=self.activity_since,
                device_time=utcnow(),
                uptime_seconds=round(self.clock() - self.started, 1),
                components={"camera": self._camera_component(), "light": self._light_component()},
                metrics=metrics,
                last_error=self.last_error,
                details=details,
            )

    def close(self):
        with self.lock:
            self.closed = True
            self.claim = None
        for part in (self.camera, self.light):
            if part.ok:
                try:
                    part.device.close()
                except Exception as exc:
                    LOG.warning("error closing %s: %s", type(part.device).__name__, exc)
