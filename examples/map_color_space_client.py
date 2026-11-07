"""
Sweep the LED light through the (r, g, b) color space with a given interval on each
dimension and take a photo at every color. N = 256 / interval steps per channel,
so N^3 photos are saved to ~/Pictures/pizerocam/color-mapping/.

Usage: python map_color_space_client.py --ip 100.64.1.100 [--interval 32]
"""
import argparse
from pathlib import Path

from image_req_client import ImageReqClient


def color_mapping_session(client, interval, settle=2.0):
    if interval not in [2, 4, 8, 16, 32, 64, 128, 256]:
        raise ValueError("interval must be a power of 2 between 2 and 256")
    out_dir = Path.home() / "Pictures" / "pizerocam" / "color-mapping"
    levels = range(interval - 1, 256, interval)
    with client.control():  # one claim for the whole sweep
        for r in levels:
            for g in levels:
                for b in levels:
                    if client.light_on(color=(r, g, b)) is None:
                        return
                    path = client.request_photo(settle=settle, output_dir=out_dir / f"{r:03d}{g:03d}{b:03d}")
                    print(f"({r},{g},{b}) -> {path}")
        client.light_off()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ip", required=True)
    parser.add_argument("--port", type=int, default=2222)
    parser.add_argument("--interval", type=int, default=32)
    args = parser.parse_args()

    client = ImageReqClient(port=args.port)
    if client.connect(args.ip):
        color_mapping_session(client, args.interval)
