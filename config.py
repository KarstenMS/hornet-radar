"""
Configuration file for Hornet Radar.
Contains constants used throughout the project.
"""
import os

# --- Raspberry Pi setup ---
PI_ID = "PI_03"

LATITUDE = 48.15423219783116     # Get the values from Google maps
LONGITUDE = 11.459030279093017

SHOW_DEBUG_VIDEO = True             # Shows Debug Video on the PI (requires GUI), default False

# --- Directories ---
ROOT = "/home/hornet/hornet-radar"

MODEL_DIR = os.path.join(ROOT, "model", "yolov5s-all-data.pt")
YOLO_DIR = os.path.join(ROOT, "yolov5")
MODEL_NAME = "yolo5"

IMAGES_DIR = os.path.join(ROOT, "detections", "images")     # For analysing single pictures
VIDEOS_DIR = os.path.join(ROOT, "detections","videos")      # For analysing videos
EVENTS_DIR = os.path.join(ROOT, "detections", "events")     # Directory for storing local events

# --- Supabase ---
SUPABASE_URL = "https://lebtnjdpjntaqheahjoi.supabase.co"
SUPABASE_KEY = "sb_secret_P7lY71HribtJdN1kIBw-Fw_6bCRZX50"
BUCKET_NAME = "hornet-detections"
TABLE_NAME = "sightings"

# --- Motion_Gate Tracking settings ---
TRACKER_TYPE = "AUTO"           # Available: "KCF", "CSRT", "MOSSE", "AUTO"
FRAME_SKIP = 3                  # analyse only every 3rd frame (performance)
TRACKING_STABLE_FRAMES = 5      # Number of frames to be stable before running detection

# --- Tracker Geometry Abort Thresholds (percent of frame) ---

TRACKER_INIT_MAX_AREA_RATIO = 0.15  # 15% vom Frame
TRACKER_MAX_AREA_RATIO = 0.35      # >35% vom Frame = unmöglich für Hornisse
TRACKER_MIN_AREA_RATIO = 0.0005    # <0.05% = Rauschen / Drift
TRACKER_MAX_ASPECT_RATIO = 5.0     # extrem langgezogen
TRACKER_MIN_ASPECT_RATIO = 0.2
TRACKER_EDGE_MARGIN_RATIO = 0.02   # 2% Randtoleranz
TRACKER_TIMEOUT = 5.0           # Seconds without update → reset

# --- Tracker ↔ Detection consistency (Abort Criterion 2) ---
TRACK_DET_MIN_IOU = 0.1                    # IoU below → inconsistent
TRACK_DET_MAX_CENTER_DIST_RATIO = 0.05     # Relative to frame diagonal
DETECTION_MIN_AREA_RATIO = 0.00005         # Ignore tiny YOLO detections


MOTION_HISTORY = 300            # Amount of frames used for Backgroundmodel ( low = faster but vulnerable to noise, high = slower, but stable background)
MOTION_VAR_THRESHOLD = 25       # Sensibility of motion detection
MOTION_MIN_AREA = 500           # Min pixel ara for being relevant

MOTION_KERNEL_SIZE = 3          # Size of morphological filtering 
MOTION_TRACK_LOST_TIMEOUT = 3.0 # Seconds, after which tracking stopps, if no motion. 

# --- Camera configuration ---
CAMERA_TYPE = "picamera2"       # "picamera2" | "webcam"

CAMERA_WIDTH = 2048 #1024
CAMERA_HEIGHT = 1536 #768
CAMERA_FPS = 30

# Webcam only
WEBCAM_INDEX = 0

# Picamera2 only
PICAM_FORMAT = "YUV420"  # "YUV420" for better performance, "XRGB8888" for better compatibility with OpenCV

# --- Detection Settings ---
CONFIDENCE_THRESHOLD = 0.65 # Optional: confidence threshold for detections
MAX_YOLO_ATTEMPTS = 3

# --- Save event ---
THUMB_SIZE = 192, 108           # Pixel-Size for thumbnails. Default: 192, 108 