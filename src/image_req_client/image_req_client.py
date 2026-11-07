import getpass
import json
import os
import socket
import threading
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from logger import get_logger

from .ph_grid_color_reader import ph_from_image

# Load environment variables from .env file
load_dotenv()

default_port = int(os.getenv("SERVER_PORT", 2222))


class DeviceError(RuntimeError):
    """A refusal or failure reported by the unit (status + stable code)."""

    def __init__(self, status, code, detail):
        super().__init__(f"{status} {code or ''}: {detail}".strip())
        self.status, self.code, self.detail = status, code, detail


class ImageReqClient:
    """
    Client for the PiCamera unit (STATUS_SPEC v1.2 service).

    Every control action needs a claim. Calls made outside ``with client.control():``
    claim, act and release on their own; inside it, one claim (kept alive by a
    heartbeat thread) covers every call.

    In the lab, workflows should drive the unit through the lab-skills SDK; this
    client is for bench use and scripts on the unit's own network.
    """

    def __init__(self, port=default_port, logger=None, timeout=30, owner=None):
        self.port = port
        self.timeout = timeout
        self.owner = owner or f"{getpass.getuser()}@{socket.gethostname()}"
        self.server_ip = None
        self.connected = False
        self.logger = logger or self._setup_logger()
        self._token = None
        self._session_id = None
        self._heartbeat_stop = None

    @staticmethod
    def _setup_logger():
        return get_logger("ImageReqClientLogger")

    @property
    def base_url(self):
        return f"http://{self.server_ip}:{self.port}"

    # --- transport ---------------------------------------------------------------

    def _request(self, method, path, body=None, token=None):
        """Return (bytes, headers). Raises DeviceError on an HTTP error, RuntimeError if unreachable."""
        if not self.server_ip:
            raise RuntimeError("Not connected to server. Call connect() first.")
        headers = {}
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if token:
            headers["X-Claim-Token"] = token
        req = Request(self.base_url + path, data=data, headers=headers, method=method)
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                return resp.read(), resp.headers
        except HTTPError as e:
            try:
                payload = json.loads(e.read())
            except ValueError:
                payload = {}
            raise DeviceError(e.code, payload.get("code"), payload.get("detail", e.reason)) from None
        except URLError as e:
            raise RuntimeError(f"{method} {path} failed: {e.reason}") from None

    def _json(self, method, path, body=None, token=None):
        data, _ = self._request(method, path, body, token)
        return json.loads(data) if data else None

    # --- connection & claims --------------------------------------------------------

    def connect(self, ip):
        """Check that the unit at `ip` answers. Returns True on success."""
        self.server_ip = ip
        try:
            probe = self._json("GET", "/")
            self.connected = True
            self.logger.info(f"Connected to {probe['equipment_name']} ({probe['equipment_id']}, "
                             f"protocol {probe['protocol_version']}) at {self.base_url}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to server at {ip}:{self.port}: {e}")
            self.connected = False
            return False

    def disconnect(self):
        self.release()
        self.connected = False
        self.server_ip = None
        return True

    def claim(self, ttl_s=30.0):
        """Acquire the unit's claim and keep it alive until release()."""
        if self._token:
            return
        self._session_id = str(uuid.uuid4())
        resp = self._json("POST", "/control/claim",
                          {"owner": self.owner, "session_id": self._session_id, "ttl_s": ttl_s})
        self._token = resp["claim_token"]
        stop = self._heartbeat_stop = threading.Event()
        interval = max(1.0, resp["heartbeat_interval_s"])

        def beat():
            while not stop.wait(interval):
                try:
                    self._request("POST", "/control/heartbeat", token=self._token)
                except Exception as e:
                    self.logger.error(f"Claim lost: {e}")
                    self._token = None
                    return

        threading.Thread(target=beat, daemon=True).start()

    def release(self):
        """Release the claim (best effort). Releasing never changes the light."""
        if self._heartbeat_stop:
            self._heartbeat_stop.set()
            self._heartbeat_stop = None
        token, self._token = self._token, None
        if token and self.server_ip:
            try:
                self._request("POST", "/control/release", token=token)
            except Exception as e:
                self.logger.warning(f"Release failed (the claim will expire): {e}")

    def control(self, ttl_s=30.0):
        """Context manager holding one claim for several actions."""
        client = self

        class _Claim:
            def __enter__(self):
                client.claim(ttl_s)
                return client

            def __exit__(self, *exc):
                client.release()

        return _Claim()

    def _control(self, path, body):
        if self._token:
            return self._json("POST", path, body, token=self._token)
        self.claim()
        try:
            return self._json("POST", path, body, token=self._token)
        finally:
            self.release()

    # --- reads -------------------------------------------------------------------------

    def status(self):
        """The unit's STATUS_SPEC envelope as a dict, or None on failure."""
        try:
            return self._json("GET", "/status")
        except Exception as e:
            self.logger.error(f"Error getting status: {e}")
            return None

    def stream_url(self):
        """Direct MJPEG stream URL (bench use; the dashboard goes through its camera broker)."""
        return f"{self.base_url}/stream"

    # --- photos ------------------------------------------------------------------------

    def request_photo(self, flash=False, settle=None, output_dir=None):
        """
        Take a photo on the unit and save a copy locally.

        Args:
            flash (bool): Turn the light on for the photo (then switch it back off)
            settle (float): Seconds to wait before capturing, 0-8 (server default if None)
            output_dir (str): Where to save (default: ~/Pictures/pizerocam)

        Returns:
            str or None: Path of the saved image, or None if failed
        """
        body = {"flash": bool(flash)}
        if settle is not None:
            body["settle_s"] = settle
        try:
            snap = self._control("/control/snapshot", body)
            data, _ = self._request("GET", snap["url"])
            out_dir = Path(output_dir) if output_dir else Path.home() / "Pictures" / "pizerocam"
            out_dir.mkdir(parents=True, exist_ok=True)
            img_path = out_dir / Path(snap["path"]).name
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
    
    # --- light -------------------------------------------------------------------------

    def light_on(self, color=None, brightness=None):
        """Turn the light on, optionally with color (r, g, b) and brightness (0-1). Returns state or None."""
        return self._light("/control/light/on", color, brightness)

    def light_off(self):
        """Turn the light off. Returns state or None."""
        return self._light("/control/light/off", None, None)

    def set_light(self, color=None, brightness=None):
        """Change light color (r, g, b) and/or brightness (0-1) without switching it on or off.
        Returns state or None."""
        return self._light("/control/light/set", color, brightness)

    def set_light_color(self, r, g, b, brightness=None):
        return self.set_light((r, g, b), brightness)

    def change_led_color(self, r, g, b):
        """Backward-compatible alias: set the light color. Returns True on success."""
        return self.set_light_color(r, g, b) is not None

    def light_state(self):
        status = self.status()
        return (status or {}).get("details", {}).get("light")

    def _light(self, path, color, brightness):
        body = {}
        if color is not None:
            body["color"] = [int(v) for v in color]
        if brightness is not None:
            body["brightness"] = brightness
        try:
            state = self._control(path, body)
            self.logger.info(f"Light: {state}")
            return state
        except Exception as e:
            self.logger.error(f"Light request {path} failed: {e}")
            return None

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
