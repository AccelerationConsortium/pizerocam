"""Agent documentation (EQUIP_GUIDE Step B3): /agent-docs and /llms.txt.

/docs and /openapi.json come from FastAPI. Links in llms.txt are relative so they
resolve under the dashboard documentation proxy's mount prefix.
"""

from importlib.resources import files

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

router = APIRouter(tags=["documentation"])


class MarkdownResponse(PlainTextResponse):
    media_type = "text/markdown"


@router.get("/agent-docs", response_class=MarkdownResponse, summary="Agent guide (Markdown)")
def agent_docs() -> str:
    return files("image_server").joinpath("docs", "AGENT_GUIDE.md").read_text(encoding="utf-8")


@router.get("/llms.txt", response_class=PlainTextResponse, summary="Discovery index for agents")
def llms_txt() -> str:
    return (
        "# PiCamera unit (STATUS_SPEC v1.2 device service)\n\n"
        "Pi Zero 2 W camera with a NeoPixel light ring: full-resolution snapshots, a live "
        "MJPEG stream, light on/off/color/brightness.\n\n"
        "## Documentation\n\n"
        "- [Agent guide](agent-docs): primary operation, claims, refusals, error codes, "
        "what a snapshot does and does not prove.\n"
        "- [OpenAPI](openapi.json): routes and schemas, including field bounds.\n"
        "- [Swagger UI](docs)\n\n"
        "## Live status\n\n"
        "Read `GET /status` through the lab-skills SDK or the dashboard. Check "
        "allowed_actions before acting. Drive the unit only through the SDK skills "
        "(pizerocam.snapshot, pizerocam.light.on/off/set), never by composing HTTP.\n"
    )
