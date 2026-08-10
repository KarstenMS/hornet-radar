"""Hornet Radar: camera abstraction for Picamera2 or standard USB webcams."""

import logging
import cv2
from config import (
    ANALOGUE_GAIN,
    AWB_ENABLE,
    AWB_MODE,
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_TYPE,
    CAMERA_WIDTH,
    CAMERA_FULL_FOV,
    CAMERA_SCALER_CROP,
    COLOUR_GAINS,
    EXPOSURE_TIME_US,
    FOCUS_DISTANCE_CM,
    PICAM_FORMAT,
    WEBCAM_INDEX,
)

logger = logging.getLogger(__name__)

class Camera:
    """Unified frame source for either Picamera2 or a USB webcam."""

    def __init__(self) -> None:
        self.camera_type = CAMERA_TYPE
        self.cap = None
        self.picam2 = None

        if self.camera_type == "picamera2":
            self._init_picamera2()
        elif self.camera_type == "webcam":
            self._init_webcam()
        else:
            raise ValueError(f"Unknown CAMERA_TYPE: {self.camera_type}")

    def _init_webcam(self) -> None: 
        self.cap = cv2.VideoCapture(WEBCAM_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

        if not self.cap.isOpened():
            raise RuntimeError("Cannot open Webcam")

    def _init_picamera2(self) -> None:
        """Initialize a Picamera2 capture device."""
        from picamera2 import Picamera2
        import time

        self.picam2 = Picamera2()
        sensor_w, sensor_h = self.picam2.sensor_resolution
        model = self.picam2.camera_properties.get("Model", "unknown")
        available_controls = self.picam2.camera_controls

        # Center-crop the sensor to the output aspect ratio. A 16:9 sensor
        # (Camera Module 3: 4608x2592) asked for a 4:3 frame otherwise
        # produces awkward scaling/letterboxing. The IMX500 is already 4:3,
        # so this is a no-op there.
        out_aspect = CAMERA_WIDTH / CAMERA_HEIGHT
        if sensor_w / sensor_h > out_aspect:
            crop_w = int(sensor_h * out_aspect)
            crop_h = sensor_h
        else:
            crop_w = sensor_w
            crop_h = int(sensor_w / out_aspect)
        scaler_crop = (
            (sensor_w - crop_w) // 2,
            (sensor_h - crop_h) // 2,
            crop_w,
            crop_h,
        )

        # Manual override to re-centre the view on the bait if needed.
        if CAMERA_SCALER_CROP is not None:
            scaler_crop = tuple(CAMERA_SCALER_CROP)

        # Diagnostics: what the sensor offers and which crop we apply. If the
        # bait is out of frame, compare scaler_crop against the sensor size here.
        try:
            mode_sizes = [m.get("size") for m in self.picam2.sensor_modes]
        except Exception:
            mode_sizes = "unavailable"
        logger.info(
            "Camera %s: sensor=%dx%d, modes=%s, ScalerCrop=%s, full_fov=%s",
            model, sensor_w, sensor_h, mode_sizes, scaler_crop, CAMERA_FULL_FOV,
        )

        # Pin the frame duration to the requested FPS. Without this, auto-
        # exposure may pick an exposure longer than 1/FPS and silently drop the
        # capture rate (e.g. to ~2 FPS in dim light), which makes fast insects
        # jump too far between frames to be tracked.
        frame_us = int(1_000_000 / CAMERA_FPS) if CAMERA_FPS else 100_000

        controls_dict = {
            "FrameDurationLimits": (frame_us, frame_us),
            "ScalerCrop": scaler_crop,
            "AwbEnable": AWB_ENABLE,
        }

        # Exposure: manual (short = motion-frozen) if configured, else auto-
        # exposure bounded by the frame budget above.
        if EXPOSURE_TIME_US is not None:
            controls_dict["AeEnable"] = False
            controls_dict["ExposureTime"] = EXPOSURE_TIME_US
            if ANALOGUE_GAIN is not None:
                controls_dict["AnalogueGain"] = ANALOGUE_GAIN
            logger.info(
                "Camera %s: manual exposure %d us, gain=%s, %.1f FPS",
                model, EXPOSURE_TIME_US, ANALOGUE_GAIN, CAMERA_FPS,
            )
        else:
            controls_dict["AeEnable"] = True
            logger.info(
                "Camera %s: auto exposure, frame duration pinned to %d us (%.1f FPS)",
                model, frame_us, CAMERA_FPS,
            )

        if "NoiseReductionMode" in available_controls:
            controls_dict["NoiseReductionMode"] = 1  # Fast

        # Auto AWB drifts per-camera and reacts to the greenscreen background.
        # Manual ColourGains (with AWB off) renders consistently across Pis.
        if AWB_ENABLE:
            if "AwbMode" in available_controls:
                controls_dict["AwbMode"] = AWB_MODE
                logger.info("Camera %s: AWB preset mode=%d", model, AWB_MODE)
        else:
            if "ColourGains" in available_controls:
                controls_dict["ColourGains"] = COLOUR_GAINS
                logger.info(
                    "Camera %s: manual ColourGains red=%.2f blue=%.2f",
                    model, COLOUR_GAINS[0], COLOUR_GAINS[1],
                )

        # Manual focus is far more reliable than continuous AF on a fixed-
        # mount camera: continuous AF hunts whenever a hornet flies through
        # and produces blurry frames during each refocus cycle. LensPosition
        # is in dioptres (1 / distance_in_metres).
        if "AfMode" in available_controls and "LensPosition" in available_controls:
            lens_position = 100.0 / max(FOCUS_DISTANCE_CM, 1)
            controls_dict["AfMode"] = 0  # Manual
            controls_dict["LensPosition"] = lens_position
            logger.info(
                "Camera %s: manual focus at %d cm (LensPosition=%.2f)",
                model, FOCUS_DISTANCE_CM, lens_position,
            )
        else:
            logger.info("Camera %s: fixed focus (no AF support)", model)

        config_kwargs = dict(
            main={
                "size": (CAMERA_WIDTH, CAMERA_HEIGHT),
                "format": PICAM_FORMAT,
            },
            controls=controls_dict,
        )
        # Read out the full sensor and scale down, so the field of view stays
        # constant regardless of the requested output resolution (otherwise
        # Picamera2 may select a cropped mode and the bait leaves the frame).
        if CAMERA_FULL_FOV:
            config_kwargs["raw"] = {"size": (sensor_w, sensor_h)}

        config = self.picam2.create_video_configuration(**config_kwargs)

        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(2)  # Allow AE/AWB to converge

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
        if self.cap:
            self.cap.release()
        if self.picam2:
            self.picam2.stop()
