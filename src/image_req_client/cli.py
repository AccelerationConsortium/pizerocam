#!/usr/bin/env python3
"""
Command-line interface for the PiCamera unit client.
"""

import argparse
import json
import sys

from .image_req_client import ImageReqClient, default_port, default_token


def _rgb(text):
    try:
        r, g, b = (int(v) for v in text.split(","))
        return r, g, b
    except ValueError:
        raise argparse.ArgumentTypeError("use R,G,B (e.g. 255,0,0)")


def main_client():
    parser = argparse.ArgumentParser(
        description="PiCamera unit client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pizerocam-client --ip 100.64.1.100                      # interactive session
  pizerocam-client --ip 100.64.1.100 --photo              # take a photo
  pizerocam-client --ip 100.64.1.100 --photo --flash      # photo with light as flash
  pizerocam-client --ip 100.64.1.100 --light on --color 255,0,0
  pizerocam-client --ip 100.64.1.100 --light off
  pizerocam-client --ip 100.64.1.100 --color 0,0,255      # change color only
  pizerocam-client --ip 100.64.1.100 --stream-url
        """,
    )
    parser.add_argument("--ip", required=True, help="IP address of the PiCamera unit")
    parser.add_argument("--port", type=int, default=default_port, help=f"Server port (default: {default_port})")
    parser.add_argument("--token", default=default_token, help="Bearer token (default: $PIZEROCAM_TOKEN)")
    parser.add_argument("--photo", action="store_true", help="Take a photo and save it locally")
    parser.add_argument("--flash", action="store_true", help="Use the light as a flash for --photo")
    parser.add_argument("--output-dir", help="Where to save photos (default: ~/Pictures/pizerocam)")
    parser.add_argument("--analyze", action="store_true", help="Analyze the photo for pH (with --photo)")
    parser.add_argument("--light", choices=["on", "off"], help="Turn the light on or off")
    parser.add_argument("--color", type=_rgb, metavar="R,G,B", help="Light color")
    parser.add_argument("--brightness", type=float, help="Light brightness 0-1")
    parser.add_argument("--led", type=_rgb, metavar="R,G,B", help=argparse.SUPPRESS)  # old name for --color
    parser.add_argument("--status", action="store_true", help="Print unit status")
    parser.add_argument("--stream-url", action="store_true", help="Print the live stream URL")
    args = parser.parse_args()
    color = args.color or args.led

    client = ImageReqClient(port=args.port, token=args.token)
    print(f"Connecting to http://{args.ip}:{args.port} ...")
    if not client.connect(args.ip):
        print("Failed to connect to server")
        sys.exit(1)

    acted = False
    ok = True
    if args.light == "on":
        acted = True
        ok &= client.light_on(color, args.brightness) is not None
    else:
        if args.light == "off":
            acted = True
            ok &= client.light_off() is not None
        if color or args.brightness is not None:
            acted = True
            ok &= client.set_light(color, args.brightness) is not None

    if args.photo:
        acted = True
        path = client.request_photo(flash=args.flash, output_dir=args.output_dir)
        if path:
            print(f"Photo saved to: {path}")
            if args.analyze:
                print(f"pH result: {client.analyze_photo(path)}")
        else:
            ok = False

    if args.status:
        acted = True
        print(json.dumps(client.status(), indent=2))
    if args.stream_url:
        acted = True
        print(client.stream_url())

    if not acted:
        try:
            client.interactive_session()
        except KeyboardInterrupt:
            print("\nInterrupted by user")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main_client()
