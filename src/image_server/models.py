"""Request/response bodies for the PiCamera unit's control and media endpoints.

The STATUS_SPEC envelope, probe, health and claim bodies come from
``sdl_lab_contract``; only this device's own shapes live here.
"""

from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Skill names reported in /status allowed_actions (must match the lab-skills catalog).
SKILL_SNAPSHOT = "pizerocam.snapshot"
SKILL_LIGHT_ON = "pizerocam.light.on"
SKILL_LIGHT_OFF = "pizerocam.light.off"
SKILL_LIGHT_SET = "pizerocam.light.set"

MAX_SETTLE_S = 8.0

Channel = Annotated[int, Field(ge=0, le=255)]
RGB = Annotated[list[Channel], Field(min_length=3, max_length=3, description="[r, g, b], each 0-255")]
Brightness = Annotated[float, Field(ge=0.0, le=1.0, description="0.0 (dark) to 1.0 (full)")]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LightOn(_Strict):
    """Turn the light on. Omitted fields keep their current value."""

    color: Optional[RGB] = None
    brightness: Optional[Brightness] = None


class LightSet(_Strict):
    """Change color and/or brightness without switching the light on or off."""

    color: Optional[RGB] = None
    brightness: Optional[Brightness] = None

    @model_validator(mode="after")
    def _something(self):
        if self.color is None and self.brightness is None:
            raise ValueError("provide color and/or brightness")
        return self


class LightOff(_Strict):
    pass


class SnapshotRequest(_Strict):
    flash: bool = Field(False, description="Switch the light on for this capture, then restore it.")
    settle_s: Optional[float] = Field(
        None, ge=0.0, le=MAX_SETTLE_S,
        description="Seconds to wait before capturing. Default: PhotoSettleSeconds when flash "
                    "switched the light on, otherwise 0.",
    )
    lens: Optional[Literal["main"]] = Field(None, description="Accepted for camera-API compatibility.")


class LightState(BaseModel):
    on: bool
    color: list[int]
    brightness: float


class SnapshotResponse(BaseModel):
    path: str = Field(description="Path relative to the media root")
    url: str = Field(description="Device-relative URL of the JPEG (under /media/)")
    taken_at: str
    lens: str = "main"
    width: Optional[int] = None
    height: Optional[int] = None
    bytes: Optional[int] = None


class MediaFile(BaseModel):
    name: str
    lens: str
    kind: Literal["snapshots", "recordings"]
    bytes: int
    mtime: str
    url: str
    abs_path: str = Field(description="Path relative to the media root (no host paths are exposed)")


class MediaListing(BaseModel):
    snapshots: list[MediaFile]
    recordings: list[MediaFile] = Field(default_factory=list)


class ErrorBody(BaseModel):
    detail: str
    code: Optional[str] = None
