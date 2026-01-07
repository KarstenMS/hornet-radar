"""
Local storage handling for DetectionEvents.
"""

import os
import json
import cv2
from typing import Optional

from helpers import create_thumbnail, timestamp, ensure_directories
from storage import upload_image_to_supabase, upload_json_to_supabase
from event import DetectionEvent
from config import (EVENTS_DIR, LATITUDE, LONGITUDE)

# ============================================================
# Save Event
# ============================================================

def save_event(event, frame) -> Optional[str]:
    """
    Saves a DetectionEvent and all related artifacts to disk.

    Creates:
    - original frame
    - labeled image
    - thumbnail
    - event.json

    Returns:
        Path to event directory or None on failure.
    """

    # --------------------------------------------------------
    # Prepare directories
    # --------------------------------------------------------

    event_time = timestamp().replace(":", "-")
    event_dir_name = f"{event.pi_id}_{event_time}"
    event_dir = os.path.join(EVENTS_DIR, event_dir_name)

    ensure_directories(event_dir)

    # --------------------------------------------------------
    # File paths
    # --------------------------------------------------------

    labeled_name = f"{event.pi_id}_{event.event_id}.jpg"
    thumb_name = f"{event.pi_id}_{event.event_id}_thumb.jpg"

    labeled_path = os.path.join(event_dir, labeled_name)
    thumb_path = os.path.join(event_dir, thumb_name)
    json_path = os.path.join(event_dir, "event.json")
    frame_path = os.path.join(event_dir, "frame.jpg")

   


    # --------------------------------------------------------
    # Save original frame
    # --------------------------------------------------------

    cv2.imwrite(frame_path, frame)

    # --------------------------------------------------------
    # Draw detections on copy
    # --------------------------------------------------------

    labeled = frame.copy()

    for d in event.detections:
        x1, y1, x2, y2 = d["bbox"]
        conf = f"{d['confidence']*100:.1f}%"
        label = "AH" if d["class_id"] == 1 else "EH"

        cv2.rectangle(labeled, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame,
            f"{label} {conf}",
            (x1, max(y1 - 6, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    cv2.imwrite(labeled_path, labeled)

    # --------------------------------------------------------
    # Thumbnail
    # --------------------------------------------------------

    create_thumbnail(labeled_path, thumb_path)

    # --------------------------------------------------------
    # Update event with image paths
    # --------------------------------------------------------

    event.image_path = labeled_path
    event.thumb_path = thumb_path

    # --------------------------------------------------------
    # Save JSON metadata
    # --------------------------------------------------------

    with open(json_path, "w") as f:
        json.dump(event.to_dict(), f, indent=2)

    print(f"[EVENT] Saved locally: {event_dir}")

    return event_dir

def upload_event(event: DetectionEvent) -> bool:
    """
    Uploads event media + JSON to Supabase.
    Returns True on success.
    """

    if not event.image_path or not event.thumb_path:
        raise RuntimeError("Event has no saved image paths")

    # --- Upload images ---
    image_name = os.path.basename(event.image_path)
    thumb_name = os.path.basename(event.thumb_path)

    image_url = upload_image_to_supabase(event.image_path, image_name)
    thumb_url = upload_image_to_supabase(event.thumb_path, thumb_name, is_thumb=True)

    if not image_url or not thumb_url:
        print("Upload failed: image or thumbnail")
        return False

    event.image_url = image_url
    event.thumb_url = thumb_url

    # --- Build JSON ---
    data = {
        "pi_id": event.pi_id,
        "timestamp": event.timestamp,
        "confidence": event.confidence,
        "species": [
            "asian_hornet" if d["class_id"] == 1 else "european_hornet"
            for d in event.detections
        ],
        "detections": event.detections,
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "image_url": image_url,
        "thumb_url": thumb_url,
        "tracking_bbox": event.tracking_bbox,
        "roi_bbox": event.roi_bbox,
        "tracking_frames": event.tracking_frames,
        "model": event.model_name,
    }

    return upload_json_to_supabase(data)
