# Imports
import torch
from ultralytics import YOLO
import os
import sys
import datetime
import requests
import json
import cv2
import warnings

warnings.filterwarnings("ignore", category=FutureWarning) # For removing Torch FutureWarnings

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
IMAGES_FOLDER = os.path.join(ROOT, "detections/frames")     # For testing


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

    # 1️ Load the image
    img = cv2.imread(image_path)
    if img is None:
        continue

    # 2️ Run YOLO inference
    results = model(img)
    predictions = results.pred[0]

    # 3️ Count detected hornet species
    ah_count, eh_count = 0, 0
    for p in predictions:
        if p[-1] == 1:   # class_id 1 = Asian hornet
            ah_count += 1
        elif p[-1] == 0: # class_id 0 = European hornet
            eh_count += 1

    # 4️ Create JSON payload
    now = datetime.datetime.now().isoformat()
    data = {
        "pi_id": PI_ID,
        "frame_id": f"{PI_ID}_Frame_{frame_id}",
        "timestamp": now,
        "species": "asian_hornet" if ah_count > 0 else "european_hornet",
        "approach_angle": None,
        "departure_angle": None,
        "latitude": LATITUDE,
        "longitude": LONGITUDE
    }

    # 5️ Send JSON to Supabase
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }

    # For now, comment this out while testing locally
    # response = requests.post(SUPABASE_URL, headers=headers, json=data)
    # print(response.status_code, response.text)

    # Print the JSON for verification
    print(json.dumps(data, indent=2))