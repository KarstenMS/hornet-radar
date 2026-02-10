import cv2
import time
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
        from libcamera import controls

        self.picam2 = Picamera2()

        sensor_size = self.picam2.sensor_resolution
        full_crop = (0, 0, sensor_size[0], sensor_size[1])

        config = self.picam2.create_video_configuration(
                main={
                    "size": (CAMERA_WIDTH, CAMERA_HEIGHT),
                    "format": PICAM_FORMAT
                },
                controls={
                    "FrameRate": CAMERA_FPS,
                    "ScalerCrop": full_crop,
                    "AfMode": 2,     # Continuous
                    "AfSpeed": 1,    # Fast
                    "AeEnable": True,
                    "AwbEnable": True,
                }
            )
        
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(1)


        print(self.picam2.capture_metadata().get("AfState"))

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
