# PiZeroCam

A Python package for remote camera control and pH analysis using Raspberry Pi Zero 2W.

**Key Features:**
- Remote camera control with LED illumination (default: RGB 10,10,10)
- Motor control for automated testing
- **Multi-color-space pH analysis** (RGB, LAB, HSV) with interpolation
- pH estimation to one decimal place using inverse distance weighting
- Modular client/server architecture
- Easy installation with optional dependencies
- CLI tools for both client and server
- **Tailscale integration** for secure remote access

## Architecture

PiZeroCam consists of two main modules:

- **`image_server`**: Runs on Raspberry Pi Zero 2W with camera and motor hardware
- **`image_req_client`**: Runs on any computer to request photos and analyze them

### Remote Access with Tailscale

PiZeroCam support integration with Tailscale for secure, easy remote access:

**Benefits:**
- **Secure**: End-to-end encrypted connections
- **Remote**: Access your Pi from anywhere in the world
- **Easy**: No port forwarding or firewall configuration needed
- **Persistent**: Stable connection even when Pi changes networks

**How it works:**
Server automatically detects Tailscale IP (100.x.x.x range)
Clients connect using the Tailscale IP instead of local network IP

## Installation

### Client Installation (on your computer)

Make sure you have set up the Goolge Cloud API: https://cloud.google.com/sdk/docs/install-sdk

For basic photo requesting and pH analysis:

```bash
git clone https://github.com/cyrilcaoyang/pizerocam.git
cd pizerocam
pip install -e ".[client]"
```

Alternative client options:
- `pip install ".[client-tesseract]"` - for local Tesseract OCR (requires Tesseract installation)
- `pip install .` - base installation without OCR dependencies

### Server Installation (on Raspberry Pi)

```bash
git clone https://github.com/cyrilcaoyang/pizerocam.git
cd pizerocam
python3 -m venv venv --system-site-packages
source venv/bin/activate

# Install system dependencies first (if needed)
sudo apt update
sudo apt install python3-picamera2

# Install the server package
pip install ".[server]"
```

**Note:** Use `--system-site-packages` to access system-level packages like `picamera2`.

#### Tailscale Integration

The server automatically detects and uses Tailscale IP addresses when available:

```bash
# Install Tailscale (if not already installed)
curl -fsSL https://tailscale.com/install.sh | sh

# Connect to your Tailscale network
sudo tailscale up

# Start the server - it will automatically use Tailscale IP
pizerocam-server
```

**Manual IP Override:**
```bash
# Force specific IP address
export PIZEROCAM_SERVER_IP="100.64.1.100"
pizerocam-server
```

### Developer Installation

To install both client and server components:

```bash
pip install .[client,server]
```

## Usage

### Quick Start with CLI Tools

**Start the server (on Raspberry Pi):**
```bash
# Default (tries 4608x2592 first, falls back automatically)
pizerocam-server

# Force maximum resolution
pizerocam-server --resolution max

# Use Full HD for faster processing
pizerocam-server --resolution fhd

# Disable motor if PCA9685 not available
pizerocam-server --no-motor
```

**Connect from client (on your computer):**
```bash
# Interactive session
pizerocam-client --ip 192.168.1.100

# Request photo and analyze pH
pizerocam-client --ip 192.168.1.100 --photo --analyze

# Change LED color to red
pizerocam-client --ip 192.168.1.100 --led 255,0,0

# Run motor
pizerocam-client --ip 192.168.1.100 --motor
```

### Multi-Color-Space pH Analysis

PiZeroCam analyzes pH values using **three different color spaces simultaneously** and returns all results:

- **RGB**: Standard red-green-blue color space
- **LAB**: Perceptually uniform color space (recommended for best accuracy)
- **HSV**: Hue-saturation-value color space

All pH analyses:
- Return interpolated values to **one decimal place** using inverse distance weighting
- Include distance metrics showing confidence for each color space
- Generate annotated images highlighting the analyzed region

**Example output:**
```
pH ANALYSIS RESULTS
============================================================
  RGB: pH   7.3  (distance:  15.23)
  LAB: pH   7.1  (distance:  12.45)
  HSV: pH   8.0  (distance:  18.67)
============================================================
```

The result with the **smallest distance** typically indicates the most reliable measurement. LAB color space generally provides the best results for color-based pH detection.

### Python API Usage

**Client API:**

```python
from image_req_client import ImageReqClient

# Create and connect to server
client = ImageReqClient()
client.connect("192.168.1.100")

# Request photo and analyze - returns all three color spaces
image_path = client.request_photo()
if image_path:
    ph_result = client.analyze_photo(image_path)
    # Returns: {'rgb': 7.3, 'lab': 7.1, 'hsv': 8.0, 'distances': {...}}
    print(f"RGB pH: {ph_result['rgb']:.1f}")
    print(f"LAB pH: {ph_result['lab']:.1f}")
    print(f"HSV pH: {ph_result['hsv']:.1f}")

# Control LED and motor
client.change_led_color(255, 0, 0)  # Red LED
client.run_motor()

# Disconnect
client.disconnect()

# Or use as context manager
with ImageReqClient() as client:
    client.connect("192.168.1.100")
    image_path, ph_result = client.request_and_analyze_photo()
    if ph_result != "NULL":
        # Use whichever color space you prefer
        lab_ph = ph_result['lab']  # LAB is best for color perception
        print(f"Image: {image_path}, LAB pH: {lab_ph:.1f}")
```

**Server API:**

```python
from image_server import ImageServer

# Basic start/stop with all hardware
server = ImageServer(host="0.0.0.0", port=2222)
print(f"Server IP: {server.get_ip_address()}")

# Hardware configuration options
server_no_hw = ImageServer(init_camera=False, init_motor=False)  # No hardware
server_cam_only = ImageServer(init_camera=True, init_motor=False)  # Camera only
server_motor_only = ImageServer(init_camera=False, init_motor=True)  # Motor only
server_full = ImageServer(init_camera=True, init_motor=True)  # All hardware

# Camera resolution options
server_max_res = ImageServer(resolution="max")  # Maximum 4608x2592
server_4k = ImageServer(resolution="4k")        # 4K 3840x2160
server_fhd = ImageServer(resolution="fhd")      # Full HD 1920x1080

# Start in background (non-blocking)
if server.start(background=True):
    print("Server started successfully")
    
    # Do other work while server runs...
    input("Press Enter to stop server...")
    
    # Stop server
    if server.stop():
        print("Server stopped successfully")

# Start in foreground (blocking)
server.start(background=False)  # This will block until stopped

# Context manager (automatic cleanup)
with ImageServer(init_camera=False, init_motor=False) as server:
    print(f"Server running at {server.get_ip_address()}")
    # Server runs until context exits

# Status checking
if not server.is_running():
    server.start()
print(f"Server status: {'Running' if server.is_running() else 'Stopped'}")
```

**pH Analysis Standalone:**

```python
from image_req_client import ph_from_image

# Analyze existing image - returns all three color spaces
ph_result = ph_from_image(
    "path/to/image.jpg",
    return_all_color_spaces=True,
    output_dir="same",  # Save annotated images in same folder
    interpolate=True     # Interpolate to 1 decimal place
)

if ph_result != "NULL":
    print(f"RGB pH: {ph_result['rgb']:.1f}")
    print(f"LAB pH: {ph_result['lab']:.1f}")
    print(f"HSV pH: {ph_result['hsv']:.1f}")
    print(f"Distances: {ph_result['distances']}")
```

## Server Control and Management

### Starting and Stopping the Server

The ImageServer provides comprehensive start/stop functionality with multiple usage patterns:

#### 1. Command Line Interface

**Basic server start:**
```bash
# Start with default settings (host: 0.0.0.0, port: 2222)
pizerocam-server

# Custom host and port
pizerocam-server --host 192.168.1.100 --port 3333

# Enable verbose logging
pizerocam-server --verbose
```

**Stopping the server:**
- Press `Ctrl+C` for graceful shutdown
- The CLI handles signal management automatically

#### 2. Python API Control

**Background Mode (Non-blocking):**
```python
from image_server import ImageServer

server = ImageServer()

# Start server in background thread
if server.start(background=True):
    print("Server started successfully")
    print(f"Server IP: {server.get_ip_address()}")
    
    # Your application continues running here
    # Server handles requests in the background
    
    # Stop when needed
    server.stop()
```

**Foreground Mode (Blocking):**
```python
from image_server import ImageServer

server = ImageServer()

try:
    # This will block until server is stopped
    server.start(background=False)
except KeyboardInterrupt:
    print("Server interrupted")
finally:
    server.stop()
```

**Context Manager (Automatic Cleanup):**
```python
from image_server import ImageServer

# Server automatically stops when exiting context
with ImageServer(host="0.0.0.0", port=2222) as server:
    print(f"Server running at {server.get_ip_address()}")
    input("Press Enter to stop...")
# Server automatically stopped here
```

**Status Monitoring:**
```python
from image_server import ImageServer
import time

server = ImageServer()

# Check if server is running
if not server.is_running():
    print("Starting server...")
    server.start()

# Monitor status
while server.is_running():
    print(f"Server status: Running on {server.get_ip_address()}")
    time.sleep(5)
    
    # Stop condition (example)
    if some_stop_condition():
        server.stop()
        break

print("Server stopped")
```

#### 3. Advanced Server Management

**Error Handling:**
```python
from image_server import ImageServer

server = ImageServer()

try:
    if server.start():
        print("Server started successfully")
    else:
        print("Failed to start server - check logs")
        
    # Server operations...
    
except Exception as e:
    print(f"Server error: {e}")
finally:
    if server.is_running():
        if server.stop():
            print("Server stopped gracefully")
        else:
            print("Error stopping server")
```

**Multiple Server Instances:**
```python
from image_server import ImageServer

# Create multiple servers on different ports
server1 = ImageServer(port=2222)
server2 = ImageServer(port=2223)

# Start both
server1.start(background=True)
server2.start(background=True)

print(f"Server 1: {server1.get_ip_address()}:2222")
print(f"Server 2: {server2.get_ip_address()}:2223")

# Stop both
server1.stop()
server2.stop()
```

### Server Features

- **Graceful Shutdown**: Proper cleanup of resources and connections
- **Thread Safety**: Safe concurrent client handling
- **Error Recovery**: Robust error handling and logging
- **Status Monitoring**: Real-time server status checking
- **Flexible Deployment**: Background or foreground operation modes
- **Context Management**: Automatic resource cleanup
- **Signal Handling**: Responds to system signals (Ctrl+C, SIGTERM)

## Troubleshooting

### Common Hardware Issues

#### Camera Memory Allocation Error
**Error:** `OSError: [Errno 12] Cannot allocate memory`

**Solution:** The server automatically tries different camera resolutions. If it still fails:
```python
# Use server without camera
server = ImageServer(init_camera=False, init_motor=True)
```

#### Motor I2C Communication Error  
**Error:** `OSError: [Errno 5] Input/output error` (PCA9685)

**Solutions:**
1. Check I2C is enabled: `sudo raspi-config` → Interface Options → I2C
2. Verify PCA9685 connection and address
3. Use server without motor:
```python
# Use server without motor
server = ImageServer(init_camera=True, init_motor=False)
```

#### LED/NeoPixel Issues
**Error:** LED initialization fails

**Check:**
- GPIO permissions: Add user to `gpio` group
- NeoPixel wiring and power supply
- Board pin configuration

#### Tailscale IP Detection Issues
**Problem:** Server not detecting Tailscale IP

**Solutions:**
1. **Check Tailscale status:**
   ```bash
   tailscale status
   tailscale ip
   ```

2. **Manual IP override:**
   ```bash
   export PIZEROCAM_SERVER_IP="100.x.x.x"  # Your Tailscale IP
   pizerocam-server
   ```

3. **Python API override:**
   ```python
   import os
   os.environ['PIZEROCAM_SERVER_IP'] = '100.x.x.x'
   server = ImageServer()
   ```

#### Full Hardware Not Available
**For testing/development without hardware:**
```python
# Minimal server for testing
server = ImageServer(init_camera=False, init_motor=False)
```

### Hardware Configuration Examples

```python
from image_server import ImageServer

# Development/Testing (no hardware required)
server = ImageServer(init_camera=False, init_motor=False)

# Photography only (camera + LED)
server = ImageServer(init_camera=True, init_motor=False)

# Motor testing only 
server = ImageServer(init_camera=False, init_motor=True)

# Full pH testing setup (all hardware)
server = ImageServer(init_camera=True, init_motor=True)
```

## Hardware Setup (Raspberry Pi Server)

### Required Components
- Raspberry Pi Zero 2W/WH
- Pi Camera 3
- NeoPixel ring LED
- PCA9685 PWM driver (for motor control)
- DC motor
- Optional: PiSugar for UPS functionality

### Setup Instructions

1. **Enable interfaces:**
   ```bash
   sudo raspi-config
   ```
   Enable SPI and I2C under "Interface Options"

2. **Install system dependencies:**
   ```bash
   sudo apt update
   sudo apt install python3-picamera2 portaudio19-dev
   ```

3. **Optional - Install PiSugar:**
   ```bash
   wget https://cdn.pisugar.com/release/pisugar-power-manager.sh
   bash pisugar-power-manager.sh -c release
   ```

4. **Install PiZeroCam:**
   ```bash
   git clone https://github.com/cyrilcaoyang/pizerocam.git
   cd pizerocam
   python3 -m venv venv --system-site-packages
   source venv/bin/activate
   pip install ".[server]"
   ```

5. **Start the server:**
   ```bash
   pizerocam-server
   ```

### Hardware Connections

**NeoPixel LED:**
- Connect to GPIO pin (configured in server code)
- Provides illumination for camera

**Motor (via PCA9685):**
- PWMA: Channel 0
- AIN1: Channel 1  
- AIN2: Channel 2
- I2C address: 0x40

## pH Analysis

The pH analysis tool uses Google Cloud Vision API for OCR text detection and multi-color-space analysis to determine pH values from test strips.

### Features
- **Multi-color-space analysis**: Analyzes pH using RGB, LAB, and HSV color spaces
- **Interpolated pH estimation**: Estimates pH to one decimal place using inverse distance weighting
- **OCR-based detection**: Detects pH numbers (1-12) in grid layouts using Google Cloud Vision API
- **Color block analysis**: Analyzes color blocks below detected numbers
- **Distance metrics**: Calculates Euclidean distances in each color space
- **Visual output**: Generates annotated images highlighting ROI and detected pH boxes
- **Smart number handling**: Handles multi-digit number splitting intelligently

### Color Spaces
- **RGB**: Standard red-green-blue color space
- **LAB**: Perceptually uniform color space (recommended for best color matching)
- **HSV**: Hue-saturation-value color space

### Requirements
- Google Cloud Vision API credentials
- Environment variable: `GOOGLE_APPLICATION_CREDENTIALS`
- OpenCV (cv2) for color space conversions

### Configuration
The analysis region is currently configured for coordinates (750, 1100, 150, 300). Modify in the source code as needed for your specific test strips.

### Output
All pH analyses return a dictionary containing:
```python
{
    'rgb': 7.3,      # pH value from RGB color space
    'lab': 7.1,      # pH value from LAB color space (recommended)
    'hsv': 8.0,      # pH value from HSV color space
    'distances': {   # Minimum distances to reference colors
        'rgb': 15.23,
        'lab': 12.45,
        'hsv': 18.67
    }
}
```

Annotated images are saved showing:
- Step 1: OCR text detection boxes
- Step 2: Defined color boxes
- Step 3: Highlighted target box with pH result

## API Reference

### ImageReqClient Methods

**Constructor:**
- `ImageReqClient(port=2222, logger=None)`
  - `port`: Server port (default: 2222)
  - `logger`: Optional logger instance

**Connection Methods:**
- `connect(ip)` - Connect to server (returns bool)
- `disconnect()` - Disconnect from server (returns bool)
- Context manager support (`__enter__`/`__exit__`)

**Photo Methods:**
- `request_photo()` - Request and receive photo (returns str filepath or None)
- `analyze_photo(filepath, output_dir="same")` - Analyze photo for pH
  - Returns: `dict` with `{'rgb': float, 'lab': float, 'hsv': float, 'distances': dict}` or `"NULL"`
- `request_and_analyze_photo(output_dir="same")` - Combined request and analysis
  - Returns: `tuple` of `(image_path, ph_result)`

**Hardware Control:**
- `change_led_color(r, g, b)` - Change LED color (returns bool)
- `run_motor()` - Run motor (returns bool)

**Interactive:**
- `interactive_session()` - Start interactive menu-driven session

### ImageServer Methods

**Constructor:**
- `ImageServer(host="0.0.0.0", port=2222, init_camera=True, init_motor=True, resolution=None)`
  - `host`: Host address to bind to
  - `port`: Port to listen on
  - `init_camera`: Whether to initialize camera hardware
  - `init_motor`: Whether to initialize motor hardware
  - `resolution`: Camera resolution preference ("max", "4k", "fhd", "hd", "vga")

**Methods:**
- `start(background=True)` - Start server (returns bool)
  - `background=True`: Non-blocking, runs in background thread
  - `background=False`: Blocking, runs in current thread
- `stop()` - Stop server gracefully (returns bool)
- `is_running()` - Check if server is currently running (returns bool)
- `get_ip_address()` - Get server's IP address (returns str)
- `get_logger()` - Get server's logger instance
- Context manager support (`__enter__`/`__exit__`)

### CLI Options

**pizerocam-client:**
```
--ip IP          Server IP address (required)
--port PORT      Server port (default: 2222)
--photo          Request photo
--analyze        Analyze photo for pH (returns RGB, LAB, HSV)
--led R,G,B      Change LED color
--motor          Run motor
--interactive    Interactive session (default)
```

**pizerocam-server:**
```
--host HOST         Bind interface (default: 0.0.0.0)
--port PORT         Listen port (default: 2222)
--verbose           Enable verbose logging
--no-camera         Disable camera initialization
--no-motor          Disable motor initialization
--resolution RES    Camera resolution (max=4608x2592, 4k=3840x2160, 
                    fhd=1920x1080, hd=1280x720, vga=640x480)
```

## Development

### Project Structure
```
pizerocam/
├── src/
│   ├── image_req_client/          # Client module
│   │   ├── __init__.py
│   │   ├── image_req_client.py    # Main client class (use this!)
│   │   ├── ph_grid_color_reader.py # Multi-color-space pH analysis
│   │   ├── simple_photo_analyzer.py # Simple pH analyzer (no OCR)
│   │   ├── photo_analyzer.py      # DEPRECATED: legacy analyzers
│   │   └── cli.py                 # CLI interface
│   ├── image_server/              # Server module
│   │   ├── __init__.py
│   │   ├── image_server.py        # High-level server wrapper
│   │   ├── server.py              # CameraServer (camera + LED + motor)
│   │   ├── PCA9685.py            # Motor driver
│   │   └── cli.py                # CLI interface
│   ├── logger.py                  # Logging utilities
│   └── socket_utils.py           # Socket communication helpers
├── examples/                      # Example scripts
│   ├── analyze_ph_multi_colorspace.py
│   ├── ph_color_space_examples.py
│   └── ph_read.py
├── pyproject.toml                # Package configuration
└── README.md
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
- GitHub Issues: https://github.com/AccelerationConsortium/pizerocam/issues
- Email: cyrilcaoyang@gmail.com
