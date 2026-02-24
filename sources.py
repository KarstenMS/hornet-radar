"""Hornet Radar: definition of supported frame input sources."""

from enum import Enum


class FrameSource(str, Enum):
    """Enumerates supported input sources for frames."""

    CAMERA = "camera"
    VIDEO = "video"
    IMAGE = "image"
