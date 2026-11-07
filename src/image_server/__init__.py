"""PiCamera unit: STATUS_SPEC v1.2 device service for a Pi Zero camera + NeoPixel light."""

__version__ = "2.0.0"


def create_app(*args, **kwargs):
    from .app import create_app as _create_app
    return _create_app(*args, **kwargs)


__all__ = ["create_app", "__version__"]
