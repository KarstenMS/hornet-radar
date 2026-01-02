"""
Utility functions for Hornet Radar project.
"""

import os
import cv2
import datetime
from config import THUMB_SIZE

def create_thumbnail(image_path, thumb_path):
    """Create a resized thumbnail version of an image."""
    img = cv2.imread(image_path)
    thumbnail = cv2.resize(img, THUMB_SIZE)
    cv2.imwrite(thumb_path, thumbnail)


def timestamp():
    """Return current ISO-formatted timestamp."""
    return datetime.datetime.now().isoformat()


def ensure_directories(*dirs):
    """Create directories if they don't exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)
