#!/usr/bin/env python3
"""
Command-line interface for the PiCamera unit server.
"""

import argparse
import signal
import sys

from .image_server import ImageServer, load_settings


def main_server():
    settings = load_settings()
    parser = argparse.ArgumentParser(
        description="PiCamera unit: HTTP server for photos, video stream and LED light",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Endpoints:
  GET       /             browser control page
  GET       /status       unit status (JSON)
  GET|POST  /photo        take photo -> JPEG   (?flash=true&settle=5)
  GET       /stream       live MJPEG video
  GET       /light        light state
  POST      /light/on     {"color": [r,g,b], "brightness": 0-1}  (both optional)
  POST      /light/off
  POST      /light/color  {"color": [r,g,b], "brightness": 0-1}

Examples:
  pizerocam-server                         # default port from server_settings.yaml
  pizerocam-server --resolution fhd        # Full HD stills
  pizerocam-server --token s3cret          # require a bearer token
  pizerocam-server --no-camera --no-light  # no hardware (testing)
  PIZEROCAM_SERVER_IP=100.x.x.x pizerocam-server
        """,
    )
    parser.add_argument("--host", default="0.0.0.0", help="Interface to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=settings["ServerPort"],
                        help=f"Port to listen on (default: {settings['ServerPort']})")
    parser.add_argument("--resolution", choices=["max", "4k", "fhd", "hd", "vga"],
                        help="Still photo resolution (default: PhotoResolution in server_settings.yaml)")
    parser.add_argument("--token", default=None,
                        help="Require this bearer token on every request (default: $PIZEROCAM_TOKEN, else no auth)")
    parser.add_argument("--no-camera", action="store_true", help="Do not initialize the camera")
    parser.add_argument("--no-light", action="store_true", help="Do not initialize the LED ring")
    args = parser.parse_args()

    # Turn SIGTERM (e.g. systemd stop) into a clean shutdown
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))

    server = ImageServer(
        host=args.host,
        port=args.port,
        init_camera=not args.no_camera,
        init_light=not args.no_light,
        resolution=args.resolution,
        token=args.token,
    )
    try:
        if not server.start(background=False):
            sys.exit(1)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main_server()
