"""Hornet Radar: project-wide configuration constants (paths, thresholds, camera, Supabase)."""

from __future__ import annotations

import os

# --- Raspberry Pi setup ---
PI_ID = os.getenv("PI_ID", "PI_05")
LATITUDE = float(os.getenv("LATITUDE", "48.15223219783116"))
LONGITUDE = float(os.getenv("LONGITUDE", "11.436030279093017"))
SHOW_DEBUG_VIDEO = os.getenv("SHOW_DEBUG_VIDEO", "true").lower() in {"1", "true", "yes"}

# --- Directories ---
ROOT = os.getenv("HORNTR_ROOT", "/home/hornet1/hornet-radar")
MODEL_DIR = os.path.join(ROOT, "model", "yolov5s-all-data.pt")
YOLO_DIR = os.path.join(ROOT, "yolov5")
MODEL_NAME = os.getenv("MODEL_NAME", "yolo5")
IMAGES_DIR = os.path.join(ROOT, "detections", "images")
VIDEOS_DIR = os.path.join(ROOT, "detections", "videos")
EVENTS_DIR = os.path.join(ROOT, "detections", "events")

# --- Camera configuration ---
CAMERA_TYPE = os.getenv("CAMERA_TYPE", "picamera2")  # "picamera2" or "webcam"
CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "2048"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "1536"))
CAMERA_FPS = int(os.getenv("CAMERA_FPS", "10"))

# Webcam only
WEBCAM_INDEX = int(os.getenv("WEBCAM_INDEX", "0"))

# Picamera2 only
PICAM_FORMAT = os.getenv("PICAM_FORMAT", "XRGB8888")  # XRGB8888 or RGB888

# --- Detection Settings ---
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))
MAX_YOLO_ATTEMPTS = int(os.getenv("MAX_YOLO_ATTEMPTS", "3"))

# --- Save event ---
THUMB_SIZE = (192, 108)  # default: (width, height)

# --- Vector settings ---
VECTOR_WINDOW = int(os.getenv("VECTOR_WINDOW", "5"))
VEKTOR_MIN_DISTANCE = float(os.getenv("VEKTOR_MIN_DISTANCE", "10.0"))

# --- Supabase ---
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://lebtnjdpjntaqheahjoi.supabase.co")
# SECURITY NOTE: do NOT hardcode service keys in code. Use environment variables.
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
BUCKET_NAME = os.getenv("BUCKET_NAME", "hornet-detections")
TABLE_NAME = os.getenv("TABLE_NAME", "sightings")

# --- MotionGate tracking settings ---
TRACKER_TYPE = os.getenv("TRACKER_TYPE", "AUTO")  # KCF, CSRT, MOSSE, AUTO
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "3"))  # analyze every Nth frame
TRACKING_STABLE_FRAMES = int(os.getenv("TRACKING_STABLE_FRAMES", "5"))

# --- Tracker geometry abort thresholds ---
TRACKER_INIT_MAX_AREA_RATIO = float(os.getenv("TRACKER_INIT_MAX_AREA_RATIO", "0.15"))
TRACKER_MAX_AREA_RATIO = float(os.getenv("TRACKER_MAX_AREA_RATIO", "0.35"))
TRACKER_MIN_AREA_RATIO = float(os.getenv("TRACKER_MIN_AREA_RATIO", "0.0005"))
TRACKER_MAX_ASPECT_RATIO = float(os.getenv("TRACKER_MAX_ASPECT_RATIO", "5.0"))
TRACKER_MIN_ASPECT_RATIO = float(os.getenv("TRACKER_MIN_ASPECT_RATIO", "0.2"))
TRACKER_EDGE_MARGIN_RATIO = float(os.getenv("TRACKER_EDGE_MARGIN_RATIO", "0.02"))
TRACKER_MAX_INVALID_FRAMES = int(os.getenv("TRACKER_MAX_INVALID_FRAMES", "5"))
MIN_POST_CONFIRM_FRAMES = int(os.getenv("MIN_POST_CONFIRM_FRAMES", "6"))

# --- Motion detection settings ---
MOTION_HISTORY = int(os.getenv("MOTION_HISTORY", "300"))
MOTION_VAR_THRESHOLD = int(os.getenv("MOTION_VAR_THRESHOLD", "25"))
MOTION_MIN_AREA = int(os.getenv("MOTION_MIN_AREA", "500"))
MOTION_KERNEL_SIZE = int(os.getenv("MOTION_KERNEL_SIZE", "3"))
