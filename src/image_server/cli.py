#!/usr/bin/env python3
"""Command-line entry point: run the PiCamera unit service with uvicorn."""

import argparse
import logging

from .service import load_settings


def main_server():
    settings = load_settings()
    parser = argparse.ArgumentParser(
        description="PiCamera unit: STATUS_SPEC v1.2 service for snapshots, MJPEG stream and LED light",
        epilog="Docs at http://<host>:<port>/docs and /agent-docs once running.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Interface to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=settings["ServerPort"],
                        help=f"Port to listen on (default: {settings['ServerPort']})")
    parser.add_argument("--resolution", choices=["max", "4k", "fhd", "hd", "vga"],
                        help="Still photo resolution (default: PhotoResolution in server_settings.yaml)")
    parser.add_argument("--media-dir", help=f"Where snapshots are stored (default: {settings['MediaDir']})")
    parser.add_argument("--no-camera", action="store_true", help="Do not initialize the camera")
    parser.add_argument("--no-light", action="store_true", help="Do not initialize the LED ring")
    args = parser.parse_args()

    if args.media_dir:
        settings["MediaDir"] = args.media_dir
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import uvicorn
    from .app import build_service, create_app

    app = create_app(lambda: build_service(settings, init_camera=not args.no_camera,
                                           init_light=not args.no_light, resolution=args.resolution))
    uvicorn.run(app, host=args.host, port=args.port, timeout_keep_alive=5)


if __name__ == "__main__":
    main_server()
