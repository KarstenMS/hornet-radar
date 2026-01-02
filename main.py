"""
Main pipeline: iterates through images, runs detection,
uploads results and thumbnails to Supabase.
"""

import os
import cv2
from detection import load_model, run_detection, count_species
from storage import upload_image_to_supabase, upload_json_to_supabase
from utils import create_thumbnail, timestamp, ensure_directories
from config import *

def main():
    ensure_directories(FRAMES_DIR, LABELED_FRAMES_DIR, LABELED_THUMBS_DIR)
    model = load_model()
    frame_id = 0

    for image_name in os.listdir(FRAMES_DIR):
        if not image_name.endswith(".jpeg"):
            continue

        frame_id += 1
        image_path = os.path.join(FRAMES_DIR, image_name)
        print(f"Processing {image_name} ...")

        img = cv2.imread(image_path)
        if img is None:
            continue

        # --- YOLO detection ---
        predictions = run_detection(img, model)
        ah_count, eh_count = count_species(predictions)

        # --- Skip if no hornets detected ---
        if not (ah_count or eh_count):
            print(f"No hornets detected in {image_name}, skipping upload.")
            continue

        image_name = f"{PI_ID}_Frame_{frame_id}.jpg"
        thumb_name = f"{PI_ID}_Frame_{frame_id}_thumb.jpg"
        local_image_path = os.path.join(LABELED_FRAMES_DIR, image_name)
        local_thumb_path = os.path.join(LABELED_THUMBS_DIR, thumb_name)
        cv2.imwrite(local_image_path, img)
        create_thumbnail(local_image_path, local_thumb_path)

        # Upload images
        image_url = upload_image_to_supabase(local_image_path, image_name)
        thumb_url = upload_image_to_supabase(local_thumb_path, thumb_name, is_thumb=True)

        data = {
            "pi_id": PI_ID,
            "frame_id": f"{PI_ID}_Frame_{frame_id}",
            "timestamp": timestamp(),
            "species": "asian_hornet" if ah_count > 0 else "european_hornet",
            "approach_angle": None,
            "departure_angle": None,
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "image_url": image_url,
            "thumb_url": thumb_url
        }

        if upload_json_to_supabase(data):
            print("Detection record uploaded.")
        else:
            print("JSON upload failed.")

if __name__ == "__main__":
    main()
