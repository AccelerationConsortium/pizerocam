import yaml
from time import sleep
from pathlib import Path
from server import CameraServer
from PCA9685 import PCA9685
from libcamera import controls

# Get the directory where this script is located
script_dir = Path(__file__).resolve().parent

# Open and read the YAML file
with open(script_dir / 'server_settings.yaml', 'r') as file:
    data = yaml.safe_load(file)
    buffer_size = data["BufferSize"]
    server_port = data["ServerPort"]


class PHTestServer(CameraServer):
    def __init__(self, host="0.0.0.0", port=server_port):
        super().__init__(host, port)
        self.motor_driver = self._init_motor_driver()
        self.PWMA = 0
        self.AIN1 = 1
        self.AIN2 = 2

    def _init_motor_driver(self):
        pwm = PCA9685(0x40, debug=False)
        pwm.setPWMFreq(50)
        self.logger.info("Motor driver initialized.")
        return pwm

    def apply_camera_controls(self):
        """
        Capture camera metadata to inspect current sensor values.
        Called on startup to print live sensor values to terminal.
        """
        if self.cam is None:
            self.logger.error("Camera not initialized - cannot read metadata")
            return None

        metadata = self.cam.capture_metadata()
        self.logger.info(f"Camera metadata:\n{metadata}")
        print("Camera Metadata:\n", metadata)

        # Disable auto controls and lock to manual mode
        self.cam.set_controls({
            "AwbEnable": False,
            "AeEnable":  False,
            "AfMode":    controls.AfModeEnum.Manual,
        })
        self.cam.capture_metadata()  # Let the above controls take effect

        return metadata

    def set_camera_controls(self, rg, bg, exp, gain, lens, autofocus=False):
        """
        Apply camera controls sent from client.
        autofocus=True  → continuous autofocus, ignores lens parameter
        autofocus=False → manual focus locked to lens value (default)
        """
        if autofocus:
            self.cam.set_controls({
                "AwbEnable": False,
                "AeEnable":  False,
                "AfMode":    controls.AfModeEnum.Continuous,  # Enable autofocus
                "ColourGains":  (rg, bg),
                "ExposureTime": exp,
                "AnalogueGain": gain,
            })
            self.logger.info("Camera set with autofocus ENABLED")
        else:
            self.cam.set_controls({
                "AwbEnable":    False,
                "AeEnable":     False,
                "AfMode":       controls.AfModeEnum.Manual,   # Fixed focus
                "ColourGains":  (rg, bg),
                "ExposureTime": exp,
                "AnalogueGain": gain,
                "LensPosition": lens,
            })
            self.logger.info(f"Camera set: rg={rg}, bg={bg}, exp={exp}, gain={gain}, lens={lens}")

    def run_motor(self, speed=20, duration=1, reverse=False):
        """
        Run motor A at the given speed for the given duration.
        Speed, duration, and direction are sent by the client.
        If client sends invalid params, defaults are used:
          speed=20, duration=1s, reverse=False
        Motor does nothing until a RUN_MOTOR command is received —
        it does not run automatically.
        """
        try:
            direction = "reverse" if reverse else "forward"
            self.logger.info(f"Running motor (speed={speed}, duration={duration}s, direction={direction})")

            self.motor_driver.setDutycycle(self.PWMA, speed)

            ain1, ain2 = (1, 0) if reverse else (0, 1)
            self.motor_driver.setLevel(self.AIN1, ain1)
            self.motor_driver.setLevel(self.AIN2, ain2)

            sleep(duration)

            self.motor_driver.setDutycycle(self.PWMA, 0)
            self.logger.info("Motor run complete.")

        except Exception as e:
            self.logger.error(f"Motor run failed: {e}")

    def handle_client(self, conn):
        """Handle client connection"""
        try:
            while True:
                msg = conn.recv(buffer_size).decode('utf-8').strip()
                if not msg:
                    break
                self.logger.info(f"Received message: {msg}")

                if msg == "TAKE_PHOTO":
                    image_path = self.take_photo()
                    if image_path:
                        self.send_photo(conn, image_path)

                elif msg == "CHANGE_COLOR":
                    conn.sendall("PLEASE SEND RGB".encode('utf-8'))
                    self.logger.info("Sent color request to client.")

                    rgb_data = conn.recv(buffer_size).decode('utf-8').strip()
                    try:
                        r, g, b = map(int, rgb_data.split(','))
                        if all(0 <= val <= 255 for val in (r, g, b)):
                            self.color = (r, g, b)
                            self.led.fill(self.color)
                            sleep(1)
                            self.led.fill((0, 0, 0))
                            conn.sendall("COLOR_CHANGED".encode('utf-8'))
                            self.logger.info(f"LED color changed to ({r}, {g}, {b})")
                        else:
                            raise ValueError("RGB values must be 0–255")
                    except Exception as e:
                        conn.sendall(f"INVALID_RGB: {e}".encode('utf-8'))
                        self.logger.error(f"Invalid RGB data: {rgb_data}")

                elif msg == "RUN_MOTOR":
                    motor_params = conn.recv(buffer_size).decode('utf-8').strip()
                    try:
                        speed, duration, reverse = motor_params.split(',')
                        speed = int(speed)
                        duration = float(duration)
                        reverse = reverse.strip().lower() == "true"
                    except Exception:
                        self.logger.warning(f"Invalid motor params '{motor_params}', using defaults")
                        speed, duration, reverse = 20, 1, False
                    self.run_motor(speed=speed, duration=duration, reverse=reverse)
                    conn.sendall("MOTOR_RUN_COMPLETE".encode('utf-8'))

                elif msg == "SET_CAMERA":
                    params = conn.recv(buffer_size).decode('utf-8').strip()
                    try:
                        rg, bg, exp, gain, lens, autofocus = params.split(',')
                        self.set_camera_controls(
                            rg=float(rg),
                            bg=float(bg),
                            exp=int(float(exp)),
                            gain=float(gain),
                            lens=float(lens),
                            autofocus=autofocus.strip().lower() == "true"
                        )
                        conn.sendall("CAMERA_SET\n".encode('utf-8'))
                    except Exception as e:
                        conn.sendall(f"CAMERA_SET_FAILED: {e}\n".encode('utf-8'))
                        self.logger.error(f"Failed to set camera controls: {e}")

                elif msg == "GET_METADATA":
                    metadata = self.apply_camera_controls()
                    if metadata:
                        metadata_str = "\n".join(f"{k}: {v}" for k, v in metadata.items())
                        conn.sendall(f"METADATA:\n{metadata_str}\n".encode('utf-8'))
                        self.logger.info("Metadata sent to client.")
                    else:
                        conn.sendall("METADATA_FAILED\n".encode('utf-8'))

        except Exception as e:
            self.logger.error(f"Handle client error: {e}")
        finally:
            conn.close()
            self.logger.info("Client connection closed.")
            self.logger.info("Waiting for new connection.")


if __name__ == "__main__":
    ph_test_server = PHTestServer()
    ph_test_server.apply_camera_controls()  # Print metadata on startup
    ph_test_server.start_server()

    # 🔒 KEEP SERVER PROCESS ALIVE
    while True:
        sleep(1)