"""
FastAPI presentation of the PiCamera unit (STATUS_SPEC v1.2 device service).

Read:    GET /  /health  /status  /stream  /media  /media/{kind}/{lens}/{name}
Docs:    GET /docs  /openapi.json  /agent-docs  /llms.txt
Claims:  POST /control/claim  /control/heartbeat  /control/release
Control: POST /control/snapshot  /control/light/on  /control/light/off  /control/light/set
         (all require X-Claim-Token; 423 without a live claim)
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import anyio
from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from sdl_lab_contract import ClaimRequest, ClaimResponse, EquipmentStatus, HealthResponse, ProbeResponse

from . import __version__
from .documentation import router as documentation_router
from .media import MediaStore
from .models import (
    ErrorBody,
    LightOff,
    LightOn,
    LightSet,
    LightState,
    MediaListing,
    SnapshotRequest,
    SnapshotResponse,
)
from .service import CameraUnitService, ControlError, Subsystem, load_settings

LOG = logging.getLogger("pizerocam")

_ERRORS = {code: {"model": ErrorBody} for code in (401, 409, 422, 423, 500, 503)}


def build_service(settings, init_camera=True, init_light=True, resolution=None):
    """Initialize hardware. A part that fails is reported in /status, not raised."""
    light = Subsystem(disabled=True)
    if init_light:
        try:
            from .hardware import Light
            device = Light(num_pixels=settings["LedCount"], color=settings["LightDefaultColor"],
                           brightness=settings["LightBrightness"], logger=LOG)
            device.blink()
            light = Subsystem(device)
        except Exception as exc:
            LOG.error("light initialization failed: %s", exc)
            light = Subsystem(error=str(exc))
    camera = Subsystem(disabled=True)
    if init_camera:
        try:
            from .hardware import Camera
            camera = Subsystem(Camera(resolution=resolution or settings["PhotoResolution"],
                                      stream_size=settings["StreamSize"], logger=LOG))
        except Exception as exc:
            LOG.error("camera initialization failed: %s", exc)
            camera = Subsystem(error=str(exc))
    media = MediaStore(settings["MediaDir"], settings["MediaMaxFiles"])
    return CameraUnitService(settings, camera, light, media)


def create_app(service_factory=None, cors_origins=None):
    """``service_factory`` builds the CameraUnitService at startup (tests inject fakes)."""
    factory = service_factory or (lambda: build_service(load_settings()))

    @asynccontextmanager
    async def lifespan(app):
        service = factory()
        app.state.service = service
        try:
            yield
        finally:
            service.close()

    app = FastAPI(
        title="PiCamera unit",
        version=__version__,
        lifespan=lifespan,
        description=(
            "Pi Zero camera unit with a NeoPixel light: full-resolution snapshots, a live MJPEG "
            "stream and light on/off/color. STATUS_SPEC v1.2. Read /agent-docs before control. "
            "Every /control action except claim requires a live X-Claim-Token (423 otherwise). "
            "No application auth: Tailnet ACLs and the dashboard edge gate access."
        ),
        responses=_ERRORS,
    )
    app.include_router(documentation_router)
    origins = cors_origins if cors_origins is not None else [
        o.strip() for o in os.getenv("PIZEROCAM_CORS_ORIGINS", "").split(",") if o.strip()]
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                           allow_headers=["Content-Type", "X-Claim-Token"])

    @app.exception_handler(ControlError)
    async def control_error(request, exc: ControlError):
        headers = {}
        if exc.extra.get("retry_after_s") is not None:
            headers["Retry-After"] = str(max(1, round(exc.extra["retry_after_s"])))
        return JSONResponse(status_code=exc.status, content=exc.body(), headers=headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        errors = [{"location": list(e["loc"]), "message": e["msg"], "type": e["type"]} for e in exc.errors()]
        return JSONResponse(status_code=422,
                            content={"detail": "invalid request", "code": "invalid_request", "errors": errors})

    def svc(request: Request) -> CameraUnitService:
        return request.app.state.service

    # --- discovery / read ------------------------------------------------------

    @app.get("/", response_model=ProbeResponse, tags=["discovery"], operation_id="probe")
    def probe(request: Request):
        return svc(request).probe()

    @app.get("/health", response_model=HealthResponse, tags=["discovery"], operation_id="health")
    def health():
        """Process liveness only; /status reports hardware faults."""
        return HealthResponse()

    @app.get("/status", response_model=EquipmentStatus, tags=["discovery"], operation_id="get_status")
    def status(request: Request):
        """Side-effect-free STATUS_SPEC v1.2 envelope."""
        return svc(request).status()

    @app.get("/stream", tags=["camera"], operation_id="stream",
             responses={200: {"content": {"multipart/x-mixed-replace": {}},
                              "description": "MJPEG stream (boundary FRAME)"}})
    async def stream(request: Request):
        """Live MJPEG video. Runs the encoder only while at least one viewer is connected.

        Read-only (no claim). On the dashboard, open it only through the authenticated
        camera broker, and only after a viewer chooses Show stream.
        """
        service = svc(request)
        camera = service._require_camera()
        await anyio.to_thread.run_sync(camera.open_stream)

        async def frames():
            seq = 0
            try:
                while not service.closed:
                    seq, frame = await anyio.to_thread.run_sync(camera.wait_frame, seq, 1.0)
                    if frame is None:
                        continue
                    yield (b"--FRAME\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
            finally:
                # Starlette cancels this generator when the viewer leaves; shield the
                # release so a cancelled scope cannot skip it and leave the encoder on.
                with anyio.CancelScope(shield=True):
                    await anyio.to_thread.run_sync(camera.close_stream)

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=FRAME",
                                 headers={"Cache-Control": "no-store"})

    @app.get("/media", response_model=MediaListing, tags=["media"], operation_id="list_media")
    def list_media(request: Request):
        """Snapshots stored on the unit, newest first (capped; oldest are deleted)."""
        return MediaListing(snapshots=svc(request).media.listing())

    @app.get("/media/{kind}/{lens}/{name}", tags=["media"], operation_id="get_media",
             responses={200: {"content": {"image/jpeg": {}}}, 404: {"model": ErrorBody}})
    def get_media(kind: str, lens: str, name: str, request: Request):
        path = svc(request).media.resolve(kind, lens, name)
        if path is None:
            return JSONResponse(status_code=404, content={"detail": "no such media file", "code": "not_found"})
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})

    # --- claims ----------------------------------------------------------------

    @app.post("/control/claim", response_model=ClaimResponse, tags=["claims"], operation_id="acquire_claim")
    def claim(body: ClaimRequest, request: Request):
        """Acquire exclusive control (ttl clamped to 5-300 s). Same owner+session is idempotent.

        Claims serialize control only: expiry or release changes no hardware state.
        """
        return svc(request).acquire(body)

    @app.post("/control/heartbeat", response_model=ClaimResponse, tags=["claims"], operation_id="heartbeat")
    def heartbeat(request: Request, x_claim_token: Optional[str] = Header(default=None)):
        """Extend the claim. 401 means the claim is lost."""
        return svc(request).heartbeat(x_claim_token)

    @app.post("/control/release", status_code=204, tags=["claims"], operation_id="release_claim")
    def release(request: Request, x_claim_token: Optional[str] = Header(default=None)):
        """Release the claim (idempotent). The light stays as it is."""
        svc(request).release(x_claim_token)
        return Response(status_code=204)

    # --- control ---------------------------------------------------------------

    @app.post("/control/snapshot", response_model=SnapshotResponse, tags=["control"], operation_id="snapshot")
    def snapshot(request: Request, body: Optional[SnapshotRequest] = None,
                 x_claim_token: Optional[str] = Header(default=None)):
        """Capture a full-resolution JPEG, store it on the unit and return its media URL.

        `flash` switches the light on for the capture and back off afterwards (409 if the
        light is unavailable). A live stream pauses briefly while the camera switches mode.
        409 `capture_in_progress` while another snapshot runs.
        """
        return svc(request).snapshot(body or SnapshotRequest(), x_claim_token)

    @app.post("/control/light/on", response_model=LightState, tags=["control"], operation_id="light_on")
    def light_on(request: Request, body: Optional[LightOn] = None,
                 x_claim_token: Optional[str] = Header(default=None)):
        """Turn the light on, optionally setting color [r,g,b] and brightness 0-1."""
        return svc(request).light_on(body or LightOn(), x_claim_token)

    @app.post("/control/light/off", response_model=LightState, tags=["control"], operation_id="light_off")
    def light_off(request: Request, body: Optional[LightOff] = None,
                  x_claim_token: Optional[str] = Header(default=None)):
        """Turn the light off. Color and brightness are kept for the next light/on."""
        return svc(request).light_off(x_claim_token)

    @app.post("/control/light/set", response_model=LightState, tags=["control"], operation_id="light_set")
    def light_set(body: LightSet, request: Request, x_claim_token: Optional[str] = Header(default=None)):
        """Change color and/or brightness without switching the light on or off."""
        return svc(request).light_set(body, x_claim_token)

    return app
