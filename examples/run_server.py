"""
Run the PiCamera unit service on a Pi Zero 2 W/WH (same as `pizerocam-server`).
Then open http://<pi-ip>:2222/docs.
"""

from image_server.cli import main_server

if __name__ == "__main__":
    main_server()
