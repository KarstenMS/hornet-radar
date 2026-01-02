# Imports
import torch
from ultralytics import YOLO
import os
#import sys
import datetime
import requests
#import json
import cv2
import warnings

warnings.filterwarnings("ignore", category=FutureWarning) # For suppressing Torch FutureWarnings

# === CONFIGURATION ===

# General R-PI setup constants
PI_ID = "PI_01"

LATITUDE = 48.154091394201636       # Get the values from Google maps
LONGITUDE = 11.459636367430676

ROOT = "/home/hornet1/hornet-radar"


# Supabase Connection

SUPABASE_URL = "https://lebtnjdpjntaqheahjoi.supabase.co"
SUPABASE_KEY = "sb_publishable_yRnBJ6G8mN-44O_8iNKltw_J2_-899y"
BUCKET_NAME = "hornet-detections"
TABLE_NAME = "sightings"
THUMB_SIZE = 192, 108

# Initialising
MODEL_PATH = os.path.join(ROOT, "model/yolov5s-all-data.pt")
YOLO_DIR = os.path.join(ROOT, "yolov5")
FRAMES_DIR = os.path.join(ROOT, "detections/frames")     # For testing
LABLED_FRAMES_DIR = os.path.join(ROOT, "detections/labled-frames")     

model = torch.hub.load(YOLO_DIR, "custom", path=MODEL_PATH, source="local")
model.conf = 0.8 # Optional: confidence threshold for detections

# === Upload image to Supabase Storage ===
def upload_image_to_supabase(image_path, image_name):
    
    # Prepare upload URL
    if "thumb.jpg" in image_name:
        upload_url = f"{SUPABASE_URL}/storage/v1/object/{BUCKET_NAME}/thumbnails/{image_name}"
    else:
        upload_url = f"{SUPABASE_URL}/storage/v1/object/{BUCKET_NAME}/{image_name}"

    with open(image_path, "rb") as f:
        image_data = f.read()

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "image/jpeg"
    }

    response = requests.post(upload_url, headers=headers, data=image_data)

    if response.status_code in [200, 201]:
        print("Image uploaded successfully!")
        # Return the public image URL
        return f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/{image_name}"
    else:
        print("Image upload failed:", response.status_code, response.text)
        return None



# === LOOP THROUGH IMAGES ===
frame_id = 0

for image_name in os.listdir(FRAMES_DIR):
    if not image_name.endswith(".jpeg"):
        continue

    image_path = os.path.join(FRAMES_DIR, image_name)
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

    # 5 Save a local copy of the result image
    image_name = f"{PI_ID}_Frame_{frame_id}.jpg"
    thumb_name = f"{PI_ID}_Frame_{frame_id}_thumb.jpg"

    thumbnail = cv2.resize(img, THUMB_SIZE)

    local_image_path = os.path.join(LABLED_FRAMES_DIR, image_name)
    local_thumb_path = os.path.join(LABLED_FRAMES_DIR, "thumbnails", thumb_name)
    
    cv2.imwrite(local_image_path, img)
    cv2.imwrite(local_thumb_path, thumbnail)

    # Upload and get public URL
    image_url = upload_image_to_supabase(local_image_path, image_name)
    thumb_url = upload_image_to_supabase(local_thumb_path, thumb_name)

    # Add it to your JSON data
    if image_url:
        data["image_url"] = image_url
    if image_url:
        data["thumb_url"] = thumb_url    


    # 6 Send JSON to Supabase

    json_url = f"{SUPABASE_URL}/rest/v1/{TABLE_NAME}"

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(json_url, headers=headers, json=data)

    if response.status_code == 201:
        print("Successfully uploaded detection!")
    else:
        print("Upload failed:", response.status_code, response.text)

    