import os
import socket
from pathlib import Path
from time import sleep
from dotenv import load_dotenv
from logger import get_logger
from socket_utils import connect_socket, send_file_name, receive_file_name
from socket_utils import send_file_size, receive_file_size, receive_file

from .ph_grid_color_reader import ph_from_image

# Load environment variables from .env file
load_dotenv()

# Get settings from environment variables
buffer_size = int(os.getenv("BUFFER_SIZE", 2048))
chunk_size = int(os.getenv("CHUNK_SIZE", 1024))
default_port = int(os.getenv("SERVER_PORT", 2222))


class ImageReqClient:
    """
    A client for requesting images from the image server and analyzing them for pH values.
    
    This class provides a clean interface for:
    - Connecting to an image server
    - Requesting photos
    - Analyzing photos for pH values (returns all three color spaces: RGB, LAB, HSV)
    - Managing the connection lifecycle
    """
    
    def __init__(self, port=default_port, logger=None):
        """
        Initialize the ImageReqClient.
        
        Args:
            port (int): Server port to connect to
            logger: Optional logger instance
        """
        self.port = port
        self.server_ip = None
        self.socket = None
        self.connected = False
        self.logger = logger or self._setup_logger()
        
    @staticmethod
    def _setup_logger():
        """Set up the logger for the client."""
        return get_logger("ImageReqClientLogger")
    
    def connect(self, ip):
        """
        Connect to the image server at the specified IP address.
        
        Args:
            ip (str): IP address of the server
            
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.server_ip = ip
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket = connect_socket(self.socket, ip, self.port, self.logger)
            
            if self.socket is None:
                self.logger.error(f"Failed to connect to server at {ip}:{self.port}")
                return False
                
            self.connected = True
            self.logger.info(f"Successfully connected to server at {ip}:{self.port}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error connecting to server: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """
        Disconnect from the image server.
        
        Returns:
            bool: True if disconnection successful, False otherwise
        """
        try:
            if self.socket:
                self.socket.close()
                self.socket = None
            self.connected = False
            self.server_ip = None
            self.logger.info("Disconnected from server")
            return True
            
        except Exception as e:
            self.logger.error(f"Error disconnecting from server: {e}")
            return False
    
    def request_photo(self):
        """
        Request a photo from the connected server.
        
        Returns:
            str or None: Filepath of the received image, or None if failed
        """
        if not self.connected or not self.socket:
            self.logger.error("Not connected to server. Call connect() first.")
            return None
            
        try:
            # Send photo request
            self.socket.sendall("TAKE_PHOTO".encode('utf-8'))
            sleep(5)  # Wait for photo capture
            
            # Receive the photo
            success, image_path = self._receive_photo()
            
            if success:
                self.logger.info(f"Photo received and saved as {image_path}")
                return image_path
            else:
                self.logger.error("Failed to receive complete photo")
                return None
                
        except Exception as e:
            self.logger.error(f"Error requesting photo: {e}")
            return None
    
    def _receive_photo(self):
        """
        Internal method to receive photo from server.
        
        Returns:
            tuple: (success: bool, image_path: str)
        """
        try:
            # Create the photos directory in ~/Pictures/pH_photos/
            home_dir = Path.home()
            output_dir = home_dir / "Pictures" / "pH_photos"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Receive the image name and echo back to confirm
            img_name = receive_file_name(self.socket, self.logger)
            self.logger.info(f"Server sent file name {img_name}.")
            send_file_name(self.socket, img_name, self.logger)
            self.logger.info("Echoed the file name back to server.")
            img_path = output_dir / img_name

            # Receive ASCII-based file size and echo back to confirm
            file_size = receive_file_size(self.socket, self.logger)
            self.logger.info(f"Server sent file size: {file_size} bytes.")
            send_file_size(self.socket, file_size, self.logger)
            self.logger.info("Echoed the file size back to server.")

            # Now receive the actual file data in chunks, and write the file to disk
            received_data = receive_file(self.socket, file_size, chunk_size, self.logger)
            with open(img_path, "wb") as f:
                f.write(received_data)
            self.logger.info(f"File {img_name} saved to: {output_dir}")
            return True, str(img_path)
            
        except Exception as e:
            self.logger.error(f"Error receiving photo: {e}")
            return False, None
    
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
    
    def change_led_color(self, r, g, b):
        """
        Change the LED color on the server.
        
        Args:
            r (int): Red value (0-255)
            g (int): Green value (0-255)
            b (int): Blue value (0-255)
            
        Returns:
            bool: True if color change successful, False otherwise
        """
        if not self.connected or not self.socket:
            self.logger.error("Not connected to server. Call connect() first.")
            return False
            
        try:
            # Validate RGB values
            if not all(0 <= val <= 255 for val in (r, g, b)):
                self.logger.error("RGB values must be between 0-255")
                return False
            
            # Send color change request
            self.socket.sendall("CHANGE_COLOR".encode('utf-8'))
            
            # Wait for server's RGB request
            response = self.socket.recv(buffer_size).decode('utf-8').strip()
            
            if response == "PLEASE SEND RGB":
                # Send RGB values
                self.socket.sendall(f"{r},{g},{b}".encode('utf-8'))
                
                # Get response from server
                result = self.socket.recv(buffer_size).decode('utf-8').strip()
                if result == "COLOR_CHANGED":
                    self.logger.info(f"Successfully changed LED color to RGB({r},{g},{b})")
                    return True
                else:
                    self.logger.error(f"Error changing color: {result}")
                    return False
            else:
                self.logger.error(f"Unexpected server response: {response}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error changing LED color: {e}")
            return False
    
    def run_motor(self):
        """
        Run the motor on the server.
        
        Returns:
            bool: True if motor run successful, False otherwise
        """
        if not self.connected or not self.socket:
            self.logger.error("Not connected to server. Call connect() first.")
            return False
            
        try:
            self.socket.sendall("RUN_MOTOR".encode('utf-8'))
            response = self.socket.recv(buffer_size).decode('utf-8').strip()
            
            if response == "MOTOR_RUN_COMPLETE":
                self.logger.info("Motor run complete")
                return True
            else:
                self.logger.error(f"Unexpected server response: {response}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error running motor: {e}")
            return False
    
    def interactive_session(self):
        """
        Start an interactive terminal session for manual control.
        Must be connected to server first.
        """
        if not self.connected:
            self.logger.error("Not connected to server. Call connect() first.")
            return
        
        print("\n" + "="*60)
        print("Interactive PiZeroCam Session")
        print("="*60)
        
        while True:
            print("\nOptions:")
            print("  1. Request photo")
            print("  2. Request photo and analyze pH")
            print("  3. Change LED color")
            print("  4. Run motor")
            print("  5. Exit")
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == "1":
                print("\nRequesting photo...")
                image_path = self.request_photo()
                if image_path:
                    print(f"Photo saved to: {image_path}")
                else:
                    print("Failed to receive photo")
                    
            elif choice == "2":
                print("\nRequesting photo and analyzing pH...")
                image_path, ph_value = self.request_and_analyze_photo(
                    return_all_color_spaces=True,
                    output_dir="same",
                    interpolate=True,
                    print_results=True  # This will print the formatted results
                )
                if image_path:
                    print(f"\n[INFO] Photo saved to: {image_path}")
                    if ph_value != "NULL":
                        # Results already printed by analyze_photo
                        pass
                    else:
                        print("pH analysis failed")
                else:
                    print("Failed to receive photo")
                    
            elif choice == "3":
                rgb_input = input("Enter RGB values (R,G,B): ").strip()
                try:
                    r, g, b = map(int, rgb_input.split(','))
                    print(f"\nChanging LED to RGB({r},{g},{b})...")
                    if self.change_led_color(r, g, b):
                        print("LED color changed successfully!")
                    else:
                        print("Failed to change LED color")
                except ValueError:
                    print("Invalid format. Use R,G,B (e.g., 255,0,0)")
                    
            elif choice == "4":
                print("\nRunning motor...")
                if self.run_motor():
                    print("Motor run complete!")
                else:
                    print("Failed to run motor")
                    
            elif choice == "5":
                print("\nExiting interactive session...")
                break
                
            else:
                print("Invalid choice. Please enter 1-5.")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically disconnect."""
        self.disconnect()


__all__ = ['ImageReqClient'] 