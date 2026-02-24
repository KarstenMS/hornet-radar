"""Hornet Radar: local persistence and Supabase upload of <DetectionEvent> artifacts."""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import cv2

from config import EVENTS_DIR, LATITUDE, LONGITUDE
from event import DetectionEvent
from helpers import create_thumbnail, ensure_directories, timestamp
from storage import upload_image_to_supabase, upload_json_to_supabase

logger = logging.getLogger(__name__)


def save_event(event: DetectionEvent, frame) -> Optional[str]:
    """Persist a <DetectionEvent> to disk (images + JSON).

    Creates in a per-event directory:
        - frame.jpg (raw frame)
        - <pi>_<event>.jpg (labeled image)
        - <pi>_<event>_thumb.jpg (thumbnail)
        - event.json

    Args:
        event: The event to store.
        frame: The frame to save as source image (numpy array).

    Returns:
        Path to the created event directory, or None on failure.

    Raises:
        RuntimeError: If no frame is provided.
    """
    if frame is None:
        raise RuntimeError("Event has no confirmed frame")

    event_time = timestamp().replace(":", "-")
    event_dir_name = f"{event.pi_id}_{event_time}"
    event_dir = os.path.join(EVENTS_DIR, event_dir_name)
    ensure_directories(event_dir)

    labeled_name = f"{event.pi_id}_{event.event_id}.jpg"
    thumb_name = f"{event.pi_id}_{event.event_id}_thumb.jpg"

    labeled_path = os.path.join(event_dir, labeled_name)
    thumb_path = os.path.join(event_dir, thumb_name)
    json_path = os.path.join(event_dir, "event.json")
    frame_path = os.path.join(event_dir, "frame.jpg")

    # Save original frame
    cv2.imwrite(frame_path, frame)

    # Draw detections
    labeled = frame.copy()
    for d in event.detections:
        x1, y1, x2, y2 = d["bbox"]
        conf = f"{float(d.get('confidence', 0.0)) * 100:.1f}%"
        # Project-specific mapping: 1 => Asian Hornet, else European Hornet
        label = "AH" if d.get("class_id") == 1 else "EH"
        cv2.rectangle(labeled, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            labeled,
            f"{label} {conf}",
            (x1, max(y1 - 6, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    cv2.imwrite(labeled_path, labeled)
    create_thumbnail(labeled_path, thumb_path)

    event.image_path = labeled_path
    event.thumb_path = thumb_path

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(event.to_dict(), f, indent=2)

    logger.info("[EVENT] Saved locally: %s", event_dir)
    return event_dir


def upload_event(event: DetectionEvent) -> bool:
    """Upload event media + JSON to Supabase.

    Args:
        event: The event to upload. It must have been saved with <save_event> first.

    Returns:
        True if upload succeeded, otherwise False.
    """
    if not event.image_path or not event.thumb_path:
        raise RuntimeError("Event has no saved image paths")

    image_name = os.path.basename(event.image_path)
    thumb_name = os.path.basename(event.thumb_path)

    image_url = upload_image_to_supabase(event.image_path, image_name)
    thumb_url = upload_image_to_supabase(event.thumb_path, thumb_name, is_thumb=True)

    if not image_url or not thumb_url:
        logger.warning("Upload failed: image or thumbnail")
        return False

    event.image_url = image_url
    event.thumb_url = thumb_url

    data = {
        "pi_id": event.pi_id,
        "timestamp": event.timestamp,
        "confidence": event.confidence,
        "species": [
            "asian_hornet" if d.get("class_id") == 1 else "european_hornet" for d in event.detections
        ],
        "detections": event.detections,
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "image_url": image_url,
        "thumb_url": thumb_url,
        "tracking_bbox": event.tracking_bbox,
        "tracking_frames": event.tracking_frames,
        "model": event.model_name,
        "source": event.source,
        "approach_vec": event.approach_vec,
        "departure_vec": event.departure_vec,
        "dwell_time": event.dwell_time,
    }

    return upload_json_to_supabase(data)
