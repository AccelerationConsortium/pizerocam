# PiZeroCam — PiCamera Unit

A networked camera unit for the AC Organic self-driving lab: a Raspberry Pi Zero 2 W with a Pi Camera
and a 12-pixel NeoPixel light ring. It serves:

- **snapshots** at full sensor resolution (optionally using the light as a flash)
- a **live MJPEG video stream**
- **light control**: on/off, color and brightness

**This repo conforms to lab status spec v1.2**
([`ac-organic-lab/docs/STATUS_SPEC.md`](https://github.com/AccelerationConsortium/ac-organic-lab/blob/main/docs/STATUS_SPEC.md)),
including v1.1 claims with hard enforcement. On the LLE dashboard it is the **Camera Unit (LLE)**
(`lle_camera_unit`).

## Architecture

- **`image_server`** runs on the Pi: a FastAPI service over the camera and light.
- **`image_req_client`** runs on any computer: a small HTTP client and CLI for bench use, plus optional
  pH test-strip analysis tools. In the lab, workflows drive the unit through the `lab-skills` SDK instead.

## HTTP API (port 2222)

| Method | Path | Description |
|---|---|---|
| GET | `/` | Probe: `{equipment_id, equipment_name, protocol_version}` |
| GET | `/health` | Process liveness |
| GET | `/status` | STATUS_SPEC v1.2 envelope (side-effect-free) |
| GET | `/stream` | Live MJPEG (`multipart/x-mixed-replace`) |
| GET | `/media` | Stored snapshots, newest first |
| GET | `/media/snapshots/main/<name>` | One stored snapshot (JPEG) |
| POST | `/control/claim` | Acquire the claim: `{owner, session_id, ttl_s}` → `{claim_token, heartbeat_interval_s, expires_at}` |
| POST | `/control/heartbeat` | Keep the claim alive (`X-Claim-Token`) |
| POST | `/control/release` | Release the claim (idempotent) |
| POST | `/control/snapshot` | Take a photo: `{flash?, settle_s?}` → `{path, url, taken_at, lens, width, height, bytes}` |
| POST | `/control/light/on` | `{color?: [r,g,b], brightness?: 0–1}` |
| POST | `/control/light/off` | |
| POST | `/control/light/set` | Change `color` and/or `brightness` without switching on/off |
| GET | `/docs`, `/openapi.json`, `/agent-docs`, `/llms.txt` | Documentation (EQUIP_GUIDE Step B3) |

Every `/control/*` action except `claim` requires a live `X-Claim-Token` and returns **423** without one.
Claims only serialize control: **releasing or losing a claim changes no hardware state** (the light stays as
it is). The full operating semantics are in [`src/image_server/docs/AGENT_GUIDE.md`](src/image_server/docs/AGENT_GUIDE.md),
also served at `/agent-docs`.

### Status semantics

- **Primary operation (§2.3):** acquiring a still photo. `activity` is `running` from the moment a
  snapshot is accepted until the JPEG is stored, otherwise `idle`. Streaming and the light are reported in
  `components.camera` / `components.light`, not as activity. `metrics.cycles_total` counts snapshots.
- `equipment_status`: `ready`; `busy` while capturing; `degraded` when the light failed or is disabled
  (photos without flash still work) or the camera is disabled on purpose; `error` when the camera failed to
  initialize; `dry_run` with both parts disabled.
- `allowed_actions`: `pizerocam.snapshot` (camera up and idle), `pizerocam.light.on` / `.off` / `.set`
  (light up).
- Refusals: 409 `camera_unavailable` / `light_unavailable` / `capture_in_progress` / `claimed`, 422
  `invalid_request`, 423 `claim_required`. There are no §6 preconditions, so no 412s.
- `last_error.code` taxonomy: `capture_failed`, `light_write_failed`. Cleared by the next successful
  control action; refusals never set it.
- Hardware is initialized once at start; there is no startup/shutdown action.
- Snapshots stay on the Pi (default `~/pizerocam-media`, newest 200 kept). The unit is not a record
  store; copy anything that matters to BitacoraDB.

### Authentication

None in the service (STATUS_SPEC best practice 11). Tailnet ACLs and the dashboard edge gate access, and
the dashboard decides who may control the unit. Do not expose port 2222 beyond the tailnet.

## Installation

### Server (on the Raspberry Pi)

```bash
sudo raspi-config            # Interface Options → enable SPI (NeoPixel on D10)
sudo apt update
sudo apt install python3-picamera2

git clone https://github.com/AccelerationConsortium/pizerocam.git
cd pizerocam
python3 -m venv venv --system-site-packages   # needed to see the apt-installed picamera2
source venv/bin/activate
pip install ".[server]"

pizerocam-server             # http://<pi>:2222/docs
```

Requires Python 3.11+ (Raspberry Pi OS Bookworm or newer). For running it as a service on a lab device,
follow [`ac-organic-lab/docs/DEVICE_PC_SETUP.md`](https://github.com/AccelerationConsortium/ac-organic-lab/blob/main/docs/DEVICE_PC_SETUP.md).

```bash
pizerocam-server --port 2222 --resolution fhd       # still resolution: max, 4k, fhd, hd, vga
pizerocam-server --media-dir /data/pizerocam        # where snapshots are kept
pizerocam-server --no-camera --no-light             # no hardware (dry_run)
```

If the camera or light fails to initialize, the service still starts and `/status` reports why
(`error` or `degraded`). Settings live in `src/image_server/server_settings.yaml`
(equipment id/name, port, resolutions, settle time, media cap, LED defaults). Set
`PIZEROCAM_CORS_ORIGINS` (comma-separated) to allow the dashboard origin in browsers.

### Client (on your computer)

```bash
pip install -e ".[client]"            # client + pH analysis tools
pip install -e .                      # client only
```

## Client usage (bench / scripts)

```bash
pizerocam-client --ip 100.64.254.98                          # interactive session
pizerocam-client --ip 100.64.254.98 --photo --flash          # saved to ~/Pictures/pizerocam
pizerocam-client --ip 100.64.254.98 --light on --color 255,0,0 --brightness 0.5
pizerocam-client --ip 100.64.254.98 --light off
pizerocam-client --ip 100.64.254.98 --status
```

```python
from image_req_client import ImageReqClient

client = ImageReqClient(port=2222)
client.connect("100.64.254.98")
path = client.request_photo(flash=True)   # claims, captures, releases

with client.control():                    # one claim, heartbeated, for several actions
    client.light_on(color=(255, 255, 255), brightness=0.8)
    client.request_photo()
    client.light_off()
```

## Hardware

- Raspberry Pi Zero 2 W/WH
- Pi Camera Module 3 (autofocus set to continuous when supported)
- 12-pixel NeoPixel ring on **GPIO10 / board.D10 (SPI MOSI)** (this pin is required)

## Development

```bash
uv run --extra test pytest
```

Tests use a fake camera and light; `tests/fixtures/status_*.json` are STATUS_SPEC snapshot fixtures
(ready, ready with claim, busy, degraded, degraded + running, error, dry_run, cleared/active `last_error`),
each checked with `sdl_lab_contract.check_consistency`.

## Optional: pH test-strip analysis (client side)

The `client` extra includes tools that analyze a captured photo of a pH test strip in RGB, LAB and HSV
(Google Cloud Vision OCR for the reference grid). See `examples/` and `ENV_SETUP.md`.

```python
path = client.request_photo(flash=True)
result = client.analyze_photo(path, roi_config={"ROI_X": 750, "ROI_Y": 1100, "ROI_W": 150, "ROI_H": 300})
```

## License

MIT License — see LICENSE.
