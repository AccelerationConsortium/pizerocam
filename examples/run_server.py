"""
Run the PiCamera unit HTTP server on a Pi Zero 2 W/WH.
Then open http://<pi-ip>:2222/ in a browser, or use the endpoints directly.
"""

from image_server import ImageServer

if __name__ == "__main__":
    ImageServer().start(background=False)
