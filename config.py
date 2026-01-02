"""
Configuration file for Hornet Radar.
Contains constants used throughout the project.
"""
import os

# --- Raspberry Pi setup ---
PI_ID = "PI_01"

LATITUDE = 48.154091394201636       # Get the values from Google maps
LONGITUDE = 11.459636367430676

# --- Directories ---
ROOT = "/home/hornet1/hornet-radar"

MODEL_DIR = os.path.join(ROOT, "model/yolov5s-all-data.pt")
YOLO_DIR = os.path.join(ROOT, "yolov5")

FRAMES_DIR = os.path.join(ROOT, "detections/frames")     # For testing
LABELED_FRAMES_DIR = os.path.join(ROOT, "detections/labled-frames")    
LABELED_FRAMES_THUMBS_DIR = os.path.join(LABELED_FRAMES_DIR, "thumbnails")  

VIDEOS_DIR = os.path.join(ROOT, "detections/videos")     # For testing
LABELED_VIDEOS_DIR = os.path.join(ROOT, "detections/labled-videos")    
LABELED_VIDEOS_THUMBS_DIR = os.path.join(LABELED_VIDEOS_DIR, "thumbnails")  

# --- Supabase ---
SUPABASE_URL = "https://lebtnjdpjntaqheahjoi.supabase.co"
SUPABASE_KEY = "sb_secret_P7lY71HribtJdN1kIBw-Fw_6bCRZX50"
BUCKET_NAME = "hornet-detections"
TABLE_NAME = "sightings"


# --- Detection Settings ---
CONFIDENCE_THRESHOLD = 0.8 # Optional: confidence threshold for detections
THUMB_SIZE = 192, 108   #Pixel-Size for thumbnails

