import json
import os
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from logger import get_logger

from .ph_grid_color_reader import ph_from_image

# Load environment variables from .env file
load_dotenv()

default_port = int(os.getenv("SERVER_PORT", 2222))
default_token = os.getenv("PIZEROCAM_TOKEN") or None


class ImageReqClient:
    """
    HTTP client for the PiCamera unit.

    - Take photos (optionally with the LED light as a flash)
    - Turn the LED light on/off and change its color/brightness
    - Get the live MJPEG stream URL
    - Analyze photos locally for pH values (RGB, LAB, HSV)
    """

    def __init__(self, port=default_port, logger=None, token=default_token, timeout=60):
        """
        Args:
            port (int): Server port
            logger: Optional logger instance
            token (str): Bearer token, if the server was started with one
            timeout (float): Request timeout in seconds
        """
        self.port = port
        self.token = token
        self.timeout = timeout
        self.server_ip = None
        self.connected = False
        self.logger = logger or self._setup_logger()

    @staticmethod
    def _setup_logger():
        return get_logger("ImageReqClientLogger")

    @property
    def base_url(self):
        return f"http://{self.server_ip}:{self.port}"

    def _request(self, method, path, params=None, body=None):
        """Send a request; return (content_type, bytes, headers). Raises RuntimeError on failure."""
        if not self.server_ip:
            raise RuntimeError("Not connected to server. Call connect() first.")
        url = self.base_url + path
        if params:
            url += "?" + urlencode(params)
        headers = {}
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = Request(url, data=data, headers=headers, method=method)
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                return resp.headers.get("Content-Type", ""), resp.read(), resp.headers
        except HTTPError as e:
            try:
                message = json.loads(e.read()).get("error", e.reason)
            except ValueError:
                message = e.reason
            raise RuntimeError(f"{method} {path} failed ({e.code}): {message}")
        except URLError as e:
            raise RuntimeError(f"{method} {path} failed: {e.reason}")

    def _json(self, method, path, params=None, body=None):
        _, data, _ = self._request(method, path, params, body)
        return json.loads(data)

    def connect(self, ip):
        """
        Check that the unit at `ip` is reachable.

        Returns:
            bool: True if the unit answered /status
        """
        self.server_ip = ip
        try:
            status = self._json("GET", "/status")
            self.connected = True
            self.logger.info(f"Connected to {status.get('device')} {status.get('version')} at {self.base_url}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to server at {ip}:{self.port}: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """HTTP is stateless; this just forgets the server address."""
        self.connected = False
        self.server_ip = None
        return True

    def status(self):
        """Return the unit status dict, or None on failure."""
        try:
            return self._json("GET", "/status")
        except Exception as e:
            self.logger.error(f"Error getting status: {e}")
            return None

    def stream_url(self):
        """URL of the live MJPEG stream (open in a browser or cv2.VideoCapture)."""
        url = f"{self.base_url}/stream"
        return url + "?" + urlencode({"token": self.token}) if self.token else url

    def request_photo(self, flash=False, settle=None, output_dir=None):
        """
        Take a photo on the unit and save it locally.

        Args:
            flash (bool): Turn the LED light on for the photo (then restore it)
            settle (float): Seconds to wait before capturing (server default if None)
            output_dir (str): Where to save (default: ~/Pictures/pizerocam)

        Returns:
            str or None: Path of the saved image, or None if failed
        """
        params = {"flash": "true" if flash else "false"}
        if settle is not None:
            params["settle"] = settle
        try:
            _, data, headers = self._request("POST", "/photo", params)
            filename = headers.get_filename() or f"capture_{datetime.now():%Y%m%d-%H%M%S}.jpg"
            out_dir = Path(output_dir) if output_dir else Path.home() / "Pictures" / "pizerocam"
            out_dir.mkdir(parents=True, exist_ok=True)
            img_path = out_dir / Path(filename).name
            img_path.write_bytes(data)
            self.logger.info(f"Photo saved to {img_path} ({len(data)} bytes)")
            return str(img_path)
        except Exception as e:
            self.logger.error(f"Error requesting photo: {e}")
            return None

    def analyze_photo(self, filepath, output_dir="same", roi_config=None):
        """
        Analyze a photo for pH value using all three color spaces (RGB, LAB, HSV).
        
        Args:
            filepath (str): Path to the image file to analyze
            output_dir (str): Directory to save annotated images. 
                             "same" = same folder as image (default)
                             None = ~/Pictures/pH_photos/
            roi_config (dict, optional): Dict with keys 'ROI_X', 'ROI_Y', 'ROI_W', 'ROI_H' 
                                        defining the pH strip location.
            
        Returns:
            dict: {'rgb': 7.3, 'lab': 7.1, 'hsv': 8.0, 'distances': {...}}
                 or "NULL" if analysis failed
        """
        try:
            if not os.path.exists(filepath):
                self.logger.error(f"Image file not found: {filepath}")
                return "NULL"
                
            # Get all color spaces from the analysis
            ph_result = ph_from_image(filepath, 
                                     return_all_color_spaces=True,
                                     output_dir=output_dir,
                                     interpolate=True,
                                     roi_config=roi_config)
            
            if ph_result == "NULL":
                self.logger.error("pH analysis failed")
                return "NULL"
            
            # Print formatted results
            print("\n" + "="*60)
            print("pH ANALYSIS RESULTS")
            print("="*60)
            print(f"  RGB: pH {ph_result['rgb']:>5.1f}  (distance: {ph_result['distances']['rgb']:>6.2f})")
            print(f"  LAB: pH {ph_result['lab']:>5.1f}  (distance: {ph_result['distances']['lab']:>6.2f})")
            print(f"  HSV: pH {ph_result['hsv']:>5.1f}  (distance: {ph_result['distances']['hsv']:>6.2f})")
            print("="*60 + "\n")
            
            # Log to file
            self.logger.info("pH analysis results:")
            self.logger.info(f"  RGB: {ph_result['rgb']:.1f} (distance: {ph_result['distances']['rgb']:.2f})")
            self.logger.info(f"  LAB: {ph_result['lab']:.1f} (distance: {ph_result['distances']['lab']:.2f})")
            self.logger.info(f"  HSV: {ph_result['hsv']:.1f} (distance: {ph_result['distances']['hsv']:.2f})")
            
            return ph_result
            
        except Exception as e:
            self.logger.error(f"Error analyzing photo: {e}")
            return "NULL"
    
    def request_and_analyze_photo(self, output_dir="same", roi_config=None):
        """
        Convenience method to request a photo and immediately analyze it.
        
        Args:
            output_dir (str): Directory to save annotated images. 
                             "same" = same folder as image (default)
                             None = ~/Pictures/pH_photos/
            roi_config (dict, optional): Dict with keys 'ROI_X', 'ROI_Y', 'ROI_W', 'ROI_H'
        
        Returns:
            tuple: (image_path: str or None, ph_result: dict or "NULL")
                   ph_result dict contains: {'rgb': 7.3, 'lab': 7.1, 'hsv': 8.0, 'distances': {...}}
        """
        image_path = self.request_photo()
        if image_path:
            ph_value = self.analyze_photo(image_path, output_dir=output_dir, roi_config=roi_config)
            return image_path, ph_value
        else:
            return None, "NULL"
    
    def light_on(self, color=None, brightness=None):
        """Turn the light on, optionally with color (r, g, b) and brightness (0-1). Returns state or None."""
        body = {}
        if color is not None:
            body["color"] = list(color)
        if brightness is not None:
            body["brightness"] = brightness
        return self._light("/light/on", body)

    def light_off(self):
        """Turn the light off. Returns state or None."""
        return self._light("/light/off", {})

    def set_light(self, color=None, brightness=None):
        """Change light color (r, g, b) and/or brightness (0-1) without switching it on or off.
        Applied now if the light is on, else at the next light_on(). Returns state or None."""
        body = {}
        if color is not None:
            body["color"] = list(color)
        if brightness is not None:
            body["brightness"] = brightness
        return self._light("/light/color", body)

    def set_light_color(self, r, g, b, brightness=None):
        """Change light color. Returns state or None."""
        return self.set_light((r, g, b), brightness)

    def light_state(self):
        try:
            return self._json("GET", "/light")
        except Exception as e:
            self.logger.error(f"Error getting light state: {e}")
            return None

    def _light(self, path, body):
        try:
            state = self._json("POST", path, body=body)
            self.logger.info(f"Light: {state}")
            return state
        except Exception as e:
            self.logger.error(f"Light request {path} failed: {e}")
            return None

    def change_led_color(self, r, g, b):
        """Backward-compatible alias: set the light color. Returns True on success."""
        return self.set_light_color(r, g, b) is not None

    def interactive_session(self):
        """Interactive terminal session for manual control. Must be connected first."""
        if not self.connected:
            self.logger.error("Not connected to server. Call connect() first.")
            return

        print("\n" + "=" * 60)
        print("Interactive PiCamera Unit Session")
        print(f"Live stream: {self.stream_url()}")
        print("=" * 60)

        while True:
            print("\nOptions:")
            print("  1. Take photo")
            print("  2. Take photo with flash")
            print("  3. Light on")
            print("  4. Light off")
            print("  5. Change light color")
            print("  6. Status")
            print("  7. Exit")
            choice = input("\nEnter your choice (1-7): ").strip()

            if choice in ("1", "2"):
                print("\nRequesting photo...")
                path = self.request_photo(flash=(choice == "2"))
                print(f"Photo saved to: {path}" if path else "Failed to take photo")
            elif choice == "3":
                print(self.light_on())
            elif choice == "4":
                print(self.light_off())
            elif choice == "5":
                rgb_input = input("Enter RGB values (R,G,B): ").strip()
                try:
                    r, g, b = map(int, rgb_input.split(","))
                except ValueError:
                    print("Invalid format. Use R,G,B (e.g., 255,0,0)")
                    continue
                print(self.set_light_color(r, g, b))
            elif choice == "6":
                print(json.dumps(self.status(), indent=2))
            elif choice == "7":
                print("\nExiting interactive session...")
                break
            else:
                print("Invalid choice. Please enter 1-7.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


__all__ = ['ImageReqClient']
