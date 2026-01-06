"""
Configuration file for Hornet Radar.
Contains constants used throughout the project.
"""
import os

# --- Raspberry Pi setup ---
PI_ID = "PI_02"

LATITUDE = 48.953091394201636       # Get the values from Google maps
LONGITUDE = 11.929636367430676

SHOW_DEBUG_VIDEO = True             # Shows Debug Video on the PI (requires GUI), default False

# --- Directories ---
ROOT = "/home/hornet1/hornet-radar"

MODEL_DIR = os.path.join(ROOT, "model/yolov5s-all-data.pt")
YOLO_DIR = os.path.join(ROOT, "yolov5")

FRAMES_DIR = os.path.join(ROOT, "detections/frames")     # For analysing single pictures
LABELED_FRAMES_DIR = os.path.join(ROOT, "detections/labled-frames")    
LABELED_FRAMES_THUMBS_DIR = os.path.join(LABELED_FRAMES_DIR, "thumbnails")  

VIDEOS_DIR = os.path.join(ROOT, "detections/videos")     # For analysing videos
LABELED_VIDEOS_DIR = os.path.join(ROOT, "detections/labled-videos")    
LABELED_VIDEOS_THUMBS_DIR = os.path.join(LABELED_VIDEOS_DIR, "thumbnails")  

EVENTS_DIR = os.path.join(ROOT, "detections", "events")

# --- Supabase ---
SUPABASE_URL = "https://lebtnjdpjntaqheahjoi.supabase.co"
SUPABASE_KEY = "sb_secret_P7lY71HribtJdN1kIBw-Fw_6bCRZX50"
BUCKET_NAME = "hornet-detections"
TABLE_NAME = "sightings"

# --- Motion_Gate Tracking settings ---
TRACKER_TYPE = "AUTO"           # Available: "KCF", "CSRT", "MOSSE", "AUTO"
FRAME_SKIP = 3                  # analyse only every 3rd frame (performance)
TRACKER_TIMEOUT = 5.0           # Seconds without update → reset
TRACKING_STABLE_FRAMES = 5      # Number of frames to be stable before running detection

MOTION_HISTORY = 300            # Amount of frames used for Backgroundmodel ( low = faster but vulnerable to noise, high = slower, but stable background)
MOTION_VAR_THRESHOLD = 25       # Sensibility of motion detection
MOTION_MIN_AREA = 500           # Min pixel ara for being relevant

MOTION_KERNEL_SIZE = 3          # Size of morphological filtering 
MOTION_TRACK_LOST_TIMEOUT = 3.0 # Seconds, after which tracking stopps, if no motion. 

# --- Camera configuration ---
CAMERA_TYPE = "picamera2"       # "picamera2" | "webcam"

CAMERA_WIDTH = 1024
CAMERA_HEIGHT = 768
CAMERA_FPS = 30

# Webcam only
WEBCAM_INDEX = 0

# Picamera2 only
PICAM_FORMAT = "RGB888"

# --- Detection Settings ---
CONFIDENCE_THRESHOLD = 0.8 # Optional: confidence threshold for detections

# --- Save event ---
THUMB_SIZE = 192, 108           # Pixel-Size for thumbnails 