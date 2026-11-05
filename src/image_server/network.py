"""Find the IP address clients should use to reach this unit (Tailscale preferred)."""

import os
import socket
import subprocess


def _tailscale_ip_netifaces():
    try:
        import netifaces
    except ImportError:
        return None
    for interface in netifaces.interfaces():
        try:
            for addr in netifaces.ifaddresses(interface).get(netifaces.AF_INET, []):
                ip = addr.get("addr", "")
                if ip.startswith("100."):
                    return ip
        except ValueError:
            continue
    return None


def _tailscale_ip_command():
    try:
        result = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if line.strip().startswith("100."):
            return line.strip()
    return None


def _lan_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return "localhost"


def detect_ip(logger=None):
    """Return PIZEROCAM_SERVER_IP if set, else the Tailscale IP, else the LAN IP."""
    manual_ip = os.getenv("PIZEROCAM_SERVER_IP")
    if manual_ip:
        source, ip = "PIZEROCAM_SERVER_IP", manual_ip
    else:
        ip = _tailscale_ip_netifaces() or _tailscale_ip_command()
        source = "Tailscale"
        if not ip:
            source, ip = "LAN", _lan_ip()
    if logger:
        logger.info(f"Unit IP address ({source}): {ip}")
    return ip
