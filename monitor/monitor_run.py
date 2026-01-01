# Imports
import torch
from ultralytics import YOLO
import os
import sys
import datetime
import requests
import json
import cv2


# === CONFIGURATION ===

# General R-PI setup constants
PI_ID = "PI_01"

LATITUDE = 48.154091394201636       # Get the values from Google maps
LONGITUDE = 11.459636367430676

ROOT = "/home/hornet1/hornet-radar"


# Connection configuration

SUPABASE_URL = "https://hornet-radar.supabase.co/rest/v1/sightings"
SUPABASE_KEY = "sbp_d35e22fc751d0252289178a3561c9583bbcc9abc"


# Initialising
MODEL_PATH = os.path.join(ROOT, "model/yolov5s-all-data.pt")
YOLO_DIR = os.path.join(ROOT, "yolov5")
IMAGES_FOLDER = os.path.join(ROOT, "detections/frames")     #for testing


model = torch.hub.load(YOLO_DIR, "custom", path=MODEL_PATH, source="local")
model.conf = 0.8 # Optional: confidence threshold for detections


# === LOOP THROUGH IMAGES ===
frame_id = 0

for image_name in os.listdir(IMAGES_FOLDER):
    if not image_name.endswith(".jpeg"):
        continue

    image_path = os.path.join(IMAGES_FOLDER, image_name)
    frame_id += 1
    print(f"Processing {image_name} ...")