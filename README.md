# PiZeroCam — PiCamera Unit

A networked camera unit for self-driving labs, built on a Raspberry Pi Zero 2 W with a Pi Camera and a NeoPixel LED ring.
It exposes a small HTTP API to:

- **take photos** at full sensor resolution (optionally using the light as a flash)
- **stream live video** (MJPEG — opens directly in a browser, OpenCV, or any MJPEG viewer)
- **turn the light on/off** and **change its color and brightness**

It also ships a Python client/CLI, and optional client-side tools for analyzing pH test-strip photos.

## Architecture

- **`image_server`** runs on the Pi: HTTP server + camera + LED ring.
- **`image_req_client`** runs on any computer: a thin HTTP client (stdlib only) plus optional photo-analysis tools.

The server listens on port **2222** by default and automatically reports its Tailscale IP (100.x.x.x) when available, so the unit can be reached securely from anywhere on your tailnet.

## HTTP API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Browser control page (live stream, photo, light controls) |
| GET | `/status` | Unit status: camera, light, stream (JSON) |
| GET / POST | `/photo` | Take a photo → `image/jpeg` |
| GET | `/stream` | Live MJPEG video (`multipart/x-mixed-replace`) |
| GET | `/light` | Light state `{"on", "color", "brightness"}` |
| POST | `/light/on` | Turn light on. Optional `color`, `brightness` |
| POST | `/light/off` | Turn light off |
| POST | `/light/color` | Change `color` and/or `brightness` without switching on/off |

**`/photo` parameters** (query string):
- `flash=true` — switch the light on for the photo, then restore it (default `false`: photo uses the light as it currently is)
- `settle=<seconds>` — wait before capturing (0–30). Defaults to `PhotoSettleSeconds` (5 s) when `flash` switched the light on, otherwise 0

**Light parameters** can be a JSON body or query parameters:
- `color`: `[r, g, b]`, `"r,g,b"` or `"#rrggbb"` (0–255 each)
- `brightness`: `0.0`–`1.0`

Errors are returned as JSON `{"error": "..."}` with status 400 (bad parameter), 401 (bad token), 404/405, or 503 (camera/light not available).

### Examples

```bash
UNIT=http://100.64.1.100:2222

curl $UNIT/status
curl -o photo.jpg "$UNIT/photo"
curl -o photo.jpg "$UNIT/photo?flash=true&settle=3"
curl -X POST $UNIT/light/on -H 'Content-Type: application/json' -d '{"color": [255, 0, 0], "brightness": 0.5}'
curl -X POST "$UNIT/light/color?color=%2300ff00"
curl -X POST $UNIT/light/off
```

Open `$UNIT/` in a browser for the control page, or `$UNIT/stream` for just the video. With OpenCV:

```python
import cv2
cap = cv2.VideoCapture("http://100.64.1.100:2222/stream")
ok, frame = cap.read()
```

The stream runs only while at least one viewer is connected. Taking a photo pauses the stream briefly while the camera switches to full resolution.

### Authentication

By default the API is open to anyone who can reach the port (keep it on Tailscale or a trusted LAN).
To require a token, start the server with `--token <secret>` or set `PIZEROCAM_TOKEN`; clients then send
`Authorization: Bearer <secret>` or `?token=<secret>` (handy for the browser page and stream URL).

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

pizerocam-server
```

Requires Python 3.11+ (Raspberry Pi OS Bookworm or newer).

#### Tailscale

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
pizerocam-server                      # logs http://100.x.x.x:2222/
PIZEROCAM_SERVER_IP=100.64.1.100 pizerocam-server   # force the reported IP
```

### Client (on your computer)

```bash
pip install -e ".[client]"            # client + pH analysis tools
pip install -e .                      # client only (no analysis dependencies)
```

## Usage

### Server CLI

```bash
pizerocam-server                          # port from server_settings.yaml (2222)
pizerocam-server --port 8080
pizerocam-server --resolution fhd         # still resolution: max, 4k, fhd, hd, vga
pizerocam-server --token s3cret           # require a bearer token
pizerocam-server --no-camera --no-light   # no hardware (testing)
```

If the camera or light fails to initialize, the server still starts; the affected endpoints return 503 and `/status` shows the error.

### Server settings

`src/image_server/server_settings.yaml`:

```yaml
ServerPort: 2222
PhotoResolution: max          # max, 4k, fhd, hd, vga (falls back to smaller sizes)
PhotoSettleSeconds: 5         # wait after flash turns the light on
StreamSize: [1280, 720]       # MJPEG stream resolution
LedCount: 12
LightDefaultColor: [255, 255, 255]
LightBrightness: 1.0
```

### Client CLI

```bash
pizerocam-client --ip 100.64.1.100                          # interactive session
pizerocam-client --ip 100.64.1.100 --photo                  # saved to ~/Pictures/pizerocam
pizerocam-client --ip 100.64.1.100 --photo --flash
pizerocam-client --ip 100.64.1.100 --light on --color 255,0,0 --brightness 0.5
pizerocam-client --ip 100.64.1.100 --color 0,0,255          # change color only
pizerocam-client --ip 100.64.1.100 --light off
pizerocam-client --ip 100.64.1.100 --status --stream-url
```

### Python client

```python
from image_req_client import ImageReqClient

client = ImageReqClient(port=2222)            # token=... if the server requires one
client.connect("100.64.1.100")

client.light_on(color=(255, 255, 255), brightness=0.8)
path = client.request_photo()                 # or request_photo(flash=True, settle=3)
client.set_light_color(0, 0, 255)
client.light_off()
print(client.status())
print(client.stream_url())
```

### Python server

```python
from image_server import ImageServer

server = ImageServer(port=2222, resolution="fhd")
server.start(background=False)                # blocks; Ctrl+C to stop

# or in the background
with ImageServer() as server:
    print(server.get_ip_address(), server.port)
    ...
```

## Hardware

- Raspberry Pi Zero 2 W/WH
- Pi Camera Module 3 (autofocus is set to continuous when supported)
- 12-pixel NeoPixel ring on **GPIO10 / board.D10 (SPI MOSI)** — this pin is required
- Optional: PiSugar UPS

## Troubleshooting

- **`Cannot allocate memory` from the camera** — the server tries smaller still resolutions automatically; you can also pass `--resolution fhd` or lower `StreamSize`.
- **Light fails to initialize** — check SPI is enabled, the ring is on D10, its power supply, and that your user can access GPIO/SPI.
- **Wrong IP reported** — check `tailscale ip -4`, or set `PIZEROCAM_SERVER_IP`.

## Optional: pH test-strip analysis (client side)

The `client` extra includes tools that analyze a captured photo of a pH test strip using RGB, LAB and HSV color spaces
(Google Cloud Vision OCR for the reference grid, interpolated to one decimal place):

```python
path = client.request_photo(flash=True)
result = client.analyze_photo(path, roi_config={"ROI_X": 750, "ROI_Y": 1100, "ROI_W": 150, "ROI_H": 300})
# {'rgb': 7.3, 'lab': 7.1, 'hsv': 8.0, 'distances': {...}}
```

Requires `GOOGLE_APPLICATION_CREDENTIALS` (see `ENV_SETUP.md`). See `examples/` for more.

## Project structure

```
pizerocam/
├── src/
│   ├── image_server/
│   │   ├── image_server.py       # HTTP API, CameraUnit, ImageServer
│   │   ├── hardware.py           # Camera (stills + MJPEG) and Light (NeoPixel)
│   │   ├── network.py            # Tailscale / LAN IP detection
│   │   ├── server_settings.yaml
│   │   └── cli.py                # pizerocam-server
│   ├── image_req_client/
│   │   ├── image_req_client.py   # HTTP client
│   │   ├── ph_grid_color_reader.py, simple_photo_analyzer.py, photo_analyzer.py
│   │   └── cli.py                # pizerocam-client
│   └── logger.py
├── examples/
└── pyproject.toml
```

## License

MIT License — see LICENSE.

## Support

- GitHub Issues: https://github.com/AccelerationConsortium/pizerocam/issues
- Email: cyrilcaoyang@gmail.com
