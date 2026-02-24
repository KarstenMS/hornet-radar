"""Hornet Radar: small cross-module helper utilities (filesystem, timestamps, thumbnails)."""

from __future__ import annotations

import datetime as _dt
import logging
import os
from pathlib import Path
from typing import Iterable, Tuple

import cv2

from config import THUMB_SIZE

logger = logging.getLogger(__name__)


def create_thumbnail(image_path: str | os.PathLike, thumb_path: str | os.PathLike, *, size: Tuple[int, int] = THUMB_SIZE) -> None:
    """Create a resized thumbnail image.

    Args:
        image_path: Path to the input image.
        thumb_path: Path to the thumbnail output file.
        size: (width, height) in pixels.

    Raises:
        FileNotFoundError: If the source image cannot be read.
        ValueError: If the thumbnail size is invalid.
    """
    if not (isinstance(size, tuple) and len(size) == 2 and all(int(x) > 0 for x in size)):
        raise ValueError(f"Invalid thumbnail size: {size}")

    image_path = Path(image_path)
    thumb_path = Path(thumb_path)
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    thumbnail = cv2.resize(img, size)
    thumb_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(thumb_path), thumbnail)
    if not ok:
        logger.warning("Failed to write thumbnail to %s", thumb_path)


def timestamp() -> str:
    """Return the current local timestamp as an ISO-8601 string."""
    return _dt.datetime.now().isoformat(timespec="seconds")


def ensure_directories(*dirs: str | os.PathLike) -> None:
    """Create one or more directories if they do not exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
