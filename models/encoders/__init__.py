"""Per-modality encoders. Each maps ONE frame to a d-dim embedding; the beam
model flattens the (B, T) window into (B*T) frames, encodes, and reshapes back.
"""
from .camera import CameraEncoder
from .gps import GPSEncoder
from .maps import MapEncoder

__all__ = ["CameraEncoder", "GPSEncoder", "MapEncoder"]
