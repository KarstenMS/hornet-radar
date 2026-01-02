"""
Pipeline utility functions for Hornet Radar project.
"""

import cv2
import os
from helpers import create_thumbnail, timestamp
from storage import upload_image_to_supabase, upload_json_to_supabase
from config import (PI_ID, LATITUDE, LONGITUDE, LABELED_FRAMES_DIR, LABELED_FRAMES_THUMBS_DIR)


def save_and_upload_detection_frame(frame, hornets, detection_id):
    image_name = f"{PI_ID}_DET_{detection_id}.jpg"
    thumb_name = f"{PI_ID}_DET_{detection_id}_thumb.jpg"

    labeled_image_path = os.path.join(LABELED_FRAMES_DIR, image_name)
    labeled_thumb_path = os.path.join(LABELED_FRAMES_THUMBS_DIR, thumb_name)

    # --- Draw bounding boxes ---
    for p in hornets:
        x1, y1, x2, y2 = p["bbox"]
        label = "AH" if p["class_id"] == 1 else "EH"
        conf = f"{p['confidence'] * 100:.2f}%"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame,
            f"{label} {conf}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_DUPLEX,
            1.0,
            (0, 0, 255),
            1
        )

    # --- Save & thumbnail ---
    cv2.imwrite(labeled_image_path, frame)
    create_thumbnail(labeled_image_path, labeled_thumb_path)

    # --- Upload ---
    image_url = upload_image_to_supabase(labeled_image_path, image_name)
    thumb_url = upload_image_to_supabase(labeled_thumb_path, thumb_name, is_thumb=True)

    # --- JSON ---
    data = {
        "pi_id": PI_ID,
        "detection_id": detection_id,
        "timestamp": timestamp(),
        "species": [
            "asian_hornet" if p["class_id"] == 1 else "european_hornet"
            for p in hornets
        ],
        "detections": [
            {
                "species": "asian_hornet" if p["class_id"] == 1 else "european_hornet",
                "confidence": p["confidence"],
                "bbox": p["bbox"]
            }
            for p in hornets
        ],
        "approach_angle": None,
        "departure_angle": None,
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "image_url": image_url,
        "thumb_url": thumb_url
    }

    upload_json_to_supabase(data)

