"""Hornet Radar: camera abstraction for Picamera2 or standard USB webcams."""

from __future__ import annotations

import logging
from typing import Optional

import cv2

from config import (
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_TYPE,
    CAMERA_WIDTH,
    PICAM_FORMAT,
    WEBCAM_INDEX,
)

logger = logging.getLogger(__name__)


class Camera:
    """Unified frame source for either Picamera2 or a USB webcam."""

    def __init__(self) -> None:
        self.camera_type = CAMERA_TYPE
        self.cap: Optional[cv2.VideoCapture] = None
        self.picam2 = None

        if self.camera_type == "picamera2":
            self._init_picamera2()
        elif self.camera_type == "webcam":
            self._init_webcam()
        else:
            raise ValueError(f"Unknown CAMERA_TYPE: {self.camera_type}")

    def _init_webcam(self) -> None:
        """Initialize an OpenCV webcam capture device."""
        self.cap = cv2.VideoCapture(WEBCAM_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")

    def _init_picamera2(self) -> None:
        """Initialize a Picamera2 capture device."""
        from picamera2 import Picamera2
        import time

        self.picam2 = Picamera2()
        sensor_size = self.picam2.sensor_resolution
        full_crop = (0, 0, sensor_size[0], sensor_size[1])

        controls_dict = {
            "FrameRate": CAMERA_FPS,
            "ScalerCrop": full_crop,
            "AeEnable": True,
            "AwbEnable": True,
        }

        available_controls = self.picam2.camera_controls
        # Autofocus support is optional
        if "AfMode" in available_controls:
            controls_dict["AfMode"] = 2  # Continuous
            if "AfSpeed" in available_controls:
                controls_dict["AfSpeed"] = 1  # Fast

        config = self.picam2.create_video_configuration(
            main={
                "size": (CAMERA_WIDTH, CAMERA_HEIGHT),
                "format": PICAM_FORMAT,
            },
            controls=controls_dict,
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(1)

    def read(self):
        """Read a single frame.

        Returns:
            frame (numpy array) or None if no frame is available.
        """
        if self.camera_type == "webcam":
            assert self.cap is not None
            ret, frame = self.cap.read()
            return frame if ret else None

        if self.camera_type == "picamera2":
            return self.picam2.capture_array("main")

        return None

    def release(self) -> None:
        """Release underlying camera resources."""
        if self.cap is not None:
            self.cap.release()
        if self.picam2 is not None:
            self.picam2.stop()
