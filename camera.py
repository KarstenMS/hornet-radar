import cv2
from config import (CAMERA_TYPE, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, WEBCAM_INDEX, PICAM_FORMAT)

class Camera:
    def __init__(self):
        self.camera_type = CAMERA_TYPE
        self.cap = None
        self.picam2 = None

        if self.camera_type == "picamera2":
            self._init_picamera2()
        elif self.camera_type == "webcam":
            self._init_webcam()
        else:
            raise ValueError(f"Unknown CAMERA_TYPE: {self.camera_type}")

    def _init_webcam(self):
        self.cap = cv2.VideoCapture(WEBCAM_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

        if not self.cap.isOpened():
            raise RuntimeError("Cannot open Webcam")

    def _init_picamera2(self):
        from picamera2 import Picamera2
        import time

        self.picam2 = Picamera2()

        sensor_size = self.picam2.sensor_resolution
        full_crop = (0, 0, sensor_size[0], sensor_size[1])

        # Basis-Controls (always)
        controls_dict = {
            "FrameRate": CAMERA_FPS,
            "ScalerCrop": full_crop,
            "AeEnable": True,
            "AwbEnable": True,
        }

        # Get Controls and check for Autofocus support
        available_controls = self.picam2.camera_controls

        # Set Autofocus to continuous if supported, otherwise fixed focus (default)
        if "AfMode" in available_controls:
            controls_dict["AfMode"] = 2      # Continuous
            if "AfSpeed" in available_controls:
                controls_dict["AfSpeed"] = 1  # Fast

        config = self.picam2.create_video_configuration(
            main={
                "size": (CAMERA_WIDTH, CAMERA_HEIGHT),
                "format": PICAM_FORMAT,
            },
            controls=controls_dict
        )

        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(1)

    def read(self):
        """
        Frame interface
        return: frame (numpy array) or None
        """
        if self.camera_type == "webcam":
            ret, frame = self.cap.read()
            return frame if ret else None

        if self.camera_type == "picamera2":
            return self.picam2.capture_array("main")

    def release(self):
        if self.cap:
            self.cap.release()
        if self.picam2:
            self.picam2.stop()
