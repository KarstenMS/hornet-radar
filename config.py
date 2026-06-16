"""Hornet Radar: project-wide configuration constants (paths, thresholds, camera, Supabase)."""

import os

# --- Raspberry Pi setup ---
PI_ID = "PI-X"

LATITUDE = 00.0                                    # Get the values from Google maps
LONGITUDE = 00.0

SHOW_DEBUG_VIDEO = True                                         # Shows Debug Video on the PI (requires GUI), default False

# --- Directories ---
ROOT = "/home/hornet/hornet-radar"

YOLO_DIR = os.path.join(ROOT, "yolov5")
MODEL_DIR = os.path.join(ROOT, "model", "yolov5s-all-data.pt")
MODEL_NAME = "yolo5"

IMAGES_DIR = os.path.join(ROOT, "detections", "images")         # For analyzing single pictures
VIDEOS_DIR = os.path.join(ROOT, "detections","videos")          # For analyzing videos
EVENTS_DIR = os.path.join(ROOT, "detections", "events")         # Directory for storing local events

# --- Camera configuration ---
CAMERA_TYPE = "picamera2"                                       # "picamera2" | "webcam"

CAMERA_WIDTH = 2048 #1024 2048
CAMERA_HEIGHT = 1536 #768 1536
CAMERA_FPS = 10

# Webcam only
WEBCAM_INDEX = 0

# Picamera2 only
PICAM_FORMAT = "RGB888"                                         # Picamera2 naming is reversed vs numpy: "RGB888" actually yields BGR arrays, matching cv2.VideoCapture.
FOCUS_DISTANCE_CM = 20                                          # Camera Module 3: focus distance in cm to the target (e.g. hive entrance). Set per-Pi. Ignored on IMX500 (fixed focus).

# --- White balance ---
# Auto AWB drifts under a saturated background (e.g. greenscreen bait surface) and
# differs per camera unit, so manual ColourGains is the more reliable default.
AWB_ENABLE = False                                              # True = use AWB_MODE preset; False = lock to COLOUR_GAINS below.
AWB_MODE = 5                                                    # libcamera preset (used only if AWB_ENABLE=True): 0=Auto 1=Incandescent 2=Tungsten 3=Fluorescent 4=Indoor 5=Daylight 6=Cloudy
COLOUR_GAINS = (1.6, 2.0)                                       # (red_gain, blue_gain). Increase blue to cool the image (less yellow). Tune per-Pi.

# --- Detection Settings ---
CONFIDENCE_THRESHOLD = 0.95                                     # Optional: confidence threshold for detections
MAX_YOLO_ATTEMPTS = 8                                           # Max YOLO inferences per track before giving up on confirmation.
YOLO_RETRY_INTERVAL_FRAMES = 5                                  # Frames to wait between consecutive YOLO attempts (so each retry sees a meaningfully different view).

# --- Save event ---
THUMB_SIZE = 192, 108                                           # Pixel-Size for thumbnails. Default: 192, 108 
EVENT_RETENTION_DAYS = 180                                      # Number of days to keep local event data before deletion
MAX_EVENT_STORAGE_GB = 5                                        # Maximum storage for events in GB. If exceeded, oldest events will be deleted. 

# --- Vector settings ---
VECTOR_WINDOW = 5                                               # Number of frames for approach/departure vector calculation
VECTOR_MIN_DISTANCE = 10.0                                      # Minimum pixel distance for a valid vector (to filter out noise)

# --- Supabase ---
SUPABASE_URL = "https://lebtnjdpjntaqheahjoi.supabase.co"
SUPABASE_KEY = ""                                               # SECURITY NOTE: Get API key from admin@hornet-radar.com
BUCKET_NAME = "hornet-detections"
TABLE_NAME = "sightings"

# --- Motion_Gate Tracking settings ---
TRACKER_TYPE = "AUTO"                                           # Available: "KCF", "CSRT", "MOSSE", "AUTO"
FRAME_SKIP = 3                                                  # analyse only every 3rd frame (performance)
TRACKING_STABLE_FRAMES = 8                                      # Number of frames to be stable before running detection

# --- Tracker Geometry Abort Thresholds (Abort Criterion) ---

TRACKER_INIT_MAX_AREA_RATIO = 0.15                              # 15% of frame
TRACKER_MAX_AREA_RATIO = 0.35                                   # >35% of frame = to big for a hornet
TRACKER_MIN_AREA_RATIO = 0.002                                  # <0.2% = too small to be a hornet (~6300 px at 2048x1536)
TRACKER_MAX_ASPECT_RATIO = 5.0                                  # extreme wide
TRACKER_MIN_ASPECT_RATIO = 0.2                                  # extreme small  
TRACKER_EDGE_MARGIN_RATIO = 0.02                                # 2% marge from edge
TRACKER_MAX_INVALID_FRAMES = 5                                  # Max consecutive implausible frames before aborting tracking

MIN_POST_CONFIRM_FRAMES = 6                                     # e.g 6–10 

# -- Motion Settings ---
MOTION_HISTORY = 300                                            # Amount of frames used for Backgroundmodel (low = faster, high = slower)
MOTION_VAR_THRESHOLD = 40                                       # Sensibility of motion detection (higher = less sensitive to slow/small movers)
MOTION_MIN_AREA = 3000                                          # Min pixel area for being relevant (~0.1% of 2048x1536, filters ants)
MOTION_KERNEL_SIZE = 5                                          # Size of morphological filtering (larger = erodes small blobs like ants before area check)


# --- Per-Pi overrides ---
# Anything defined in config_local.py wins over the defaults above.
# config_local.py is gitignored, so per-Pi values survive `git pull`.
# See config_local.example.py for the template.
try:
    from config_local import *  # noqa: F401, F403
except ImportError:
    pass
