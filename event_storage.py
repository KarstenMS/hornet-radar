"""
Local storage handling for DetectionEvents.
"""

import os
import json
import cv2
from typing import Optional

from helpers import create_thumbnail, timestamp, ensure_directories
from config import (EVENTS_DIR)

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

    frame_path = os.path.join(event_dir, "frame.jpg")
    labeled_path = os.path.join(event_dir, "labeled.jpg")
    thumb_path = os.path.join(event_dir, "thumb.jpg")
    json_path = os.path.join(event_dir, "event.json")

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
        cls = d.get("class_id", "?")

        cv2.rectangle(labeled, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            labeled,
            f"{cls} {conf}",
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
    # Save JSON metadata
    # --------------------------------------------------------

    with open(json_path, "w") as f:
        json.dump(event.to_dict(), f, indent=2)

    print(f"[EVENT] Saved locally: {event_dir}")

    return event_dir
