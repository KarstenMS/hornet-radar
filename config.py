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
PICAM_FORMAT = "XRGB8888"                                       # XRGB8888 or RGB888

# --- Detection Settings ---
CONFIDENCE_THRESHOLD = 0.75                                     # Optional: confidence threshold for detections
MAX_YOLO_ATTEMPTS = 3

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
TRACKING_STABLE_FRAMES = 5                                      # Number of frames to be stable before running detection

# --- Tracker Geometry Abort Thresholds (Abort Criterion) ---

TRACKER_INIT_MAX_AREA_RATIO = 0.15                              # 15% of frame
TRACKER_MAX_AREA_RATIO = 0.35                                   # >35% of frame = to big for a hornet
TRACKER_MIN_AREA_RATIO = 0.0005                                 # <0.05% = Noise
TRACKER_MAX_ASPECT_RATIO = 5.0                                  # extreme wide
TRACKER_MIN_ASPECT_RATIO = 0.2                                  # extreme small  
TRACKER_EDGE_MARGIN_RATIO = 0.02                                # 2% marge from edge
TRACKER_MAX_INVALID_FRAMES = 5                                  # Max consecutive implausible frames before aborting tracking

MIN_POST_CONFIRM_FRAMES = 6                                     # e.g 6–10 

# -- Motion Settings ---
MOTION_HISTORY = 300                                            # Amount of frames used for Backgroundmodel (low = faster, high = slower)
MOTION_VAR_THRESHOLD = 25                                       # Sensibility of motion detection
MOTION_MIN_AREA = 500                                           # Min pixel ara for being relevant
MOTION_KERNEL_SIZE = 3                                          # Size of morphological filtering 


