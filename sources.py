# sources.py
from enum import Enum

class FrameSource(str, Enum):
    CAMERA = "camera"
    VIDEO = "video"
    IMAGE = "image"
