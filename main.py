"""Hornet Radar: CLI entry point that routes frames into <MotionGate> and persists events."""

import argparse
import logging
import os
import threading
import cv2
from cleanup import cleanup_events
from config import (
    CAMERA_FPS,
    CONFIDENCE_THRESHOLD,
    EVENTS_DIR,
    IMAGES_DIR,
    SHOW_DEBUG_VIDEO,
    VIDEOS_DIR,
)

from helpers import ensure_directories
from camera import Camera
from motion_gate import MotionGate
from event_storage import save_event, upload_event
from sources import FrameSource

logger = logging.getLogger(__name__)

def resolve_source(args: argparse.Namespace) -> FrameSource:
    """Resolve input source based on CLI arguments."""
    if args.images:
        return FrameSource.IMAGE
    if args.videos:
        return FrameSource.VIDEO
    return FrameSource.CAMERA


def process_images(motion_gate: MotionGate):
    """Process all .jpg files from IMAGES_DIR."""
    logger.info("Processing images from %s", IMAGES_DIR)

    for filename in os.listdir(IMAGES_DIR):
        if not filename.lower().endswith(".jpg"):
            continue

        path = os.path.join(IMAGES_DIR, filename)
        frame = cv2.imread(path)

        if frame is None:
            logger.warning("Could not read image: %s", path)
            continue

        events, debug = motion_gate.process_frame(frame, FrameSource.IMAGE)
        logger.debug("Debug: %s", debug)

        for event in events:
            if event.confidence >= CONFIDENCE_THRESHOLD:
                save_event(event, frame)
                upload_event(event)

def process_videos(motion_gate: MotionGate):
    """Process all .mp4 files from VIDEOS_DIR."""
    logger.info("Processing videos from %s", VIDEOS_DIR)

    for filename in os.listdir(VIDEOS_DIR):
        if not filename.lower().endswith(".mp4"):
            continue

        path = os.path.join(VIDEOS_DIR, filename)
        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            logger.warning("Could not open video: %s", filename)
            continue

        logger.info("Processing video: %s", filename)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            

            events, debug = motion_gate.process_frame(frame, FrameSource.VIDEO)
            logger.debug("Debug: %s", debug)

            confirmed = [e for e in events if e.confidence >= CONFIDENCE_THRESHOLD]
            if confirmed:
                for event in confirmed:
                    save_event(event, frame)
                    upload_event(event)
                break  # stop after first confirmed frame

        cap.release()

def process_camera(motion_gate: MotionGate) -> None:
    """Process frames from live camera until ESC is pressed."""
    logger.info("Capturing from camera @ %s FPS", CAMERA_FPS)

    cam = Camera()

    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue

            events, debug = motion_gate.process_frame(frame, FrameSource.CAMERA)

            for event in events:
                if event.confidence >= CONFIDENCE_THRESHOLD:
                    save_event(event, event.frame)
                    threading.Thread(target=upload_event, args=(event,), daemon=True).start()

            # --- Optional debug window ---
            if SHOW_DEBUG_VIDEO:
                display = frame.copy()
                draw_debug_overlay(display, debug)
                cv2.imshow("Hornet Debug", display)

                if cv2.waitKey(1) & 0xFF == 27: 
                    break

    finally:
        cam.release()
        cv2.destroyAllWindows()


def draw_debug_overlay(frame, debug: dict) -> None:
    """Draw a textual overlay containing debug information."""
    y = 20
    step = 22


    def line(text, color=(255, 255, 255)):
        nonlocal y
        cv2.putText(
            frame, text, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )
        y += step

    line("Press ESC to exit")
    line(f"Source: {debug.get('source')}")
    line(f"Confidence Threshold: {CONFIDENCE_THRESHOLD:.2f}")
    line(f"FPS: {debug.get('fps', 0):.1f}" if debug.get("fps") else "FPS: -")

    line(
        f"Motion: {'YES' if debug.get('motion') else 'NO'}",
        (0, 0, 255) if debug.get("motion") else (0, 255, 0)
    )

    line(
        f"Tracking: {'ACTIVE' if debug.get('tracking') else 'IDLE'}",
        (0, 255, 255) if debug.get("tracking") else (150, 150, 150)
    )

    tracks = debug.get("tracks", []) or []
    line(f"Tracks: {len(tracks)}")
    line(f"YOLO run: {'YES' if debug.get('yolo_ran') else 'NO'}")

    # Raw motion boxes (thin red) for debugging the detector.
    for (x, yb, w, h) in debug.get("motion_boxes", []) or []:
        cv2.rectangle(frame, (x, yb), (x + w, yb + h), (0, 0, 255), 1)

    # One box per active track, colored by state.
    for t in tracks:
        x, y, w, h = map(int, t["bbox"])

        if t.get("confirmed"):
            label = t.get("label", "?")
            conf = t.get("conf") or 0.0
            color = (0, 0, 255) if label == "AH" else (0, 255, 0)
            text = f"#{t['id']} {label} {conf:.2f}"
        else:
            color = (255, 200, 0)  # tracked, not yet confirmed
            text = f"#{t['id']} TRACK"

        if t.get("coasting"):
            text += " (coast)"

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x, y - th - 6), (x + tw + 4, y), color, -1)
        cv2.putText(frame, text, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

def main():
    """CLI main function."""
    parser = argparse.ArgumentParser(description="Hornet Radar main pipeline")

    parser.add_argument("-v", "--videos", default=False, action="store_true", help="Analyze .mp4 from detections/videos")
    parser.add_argument("-i", "--images", default=False, action="store_true", help="Analyze .jpg from detections/images")
    parser.add_argument("-l", "--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Third-party libraries are very chatty at DEBUG and drown out our own
    # tracking logs; keep them at WARNING regardless of our log level.
    for noisy in ("picamera2", "libcamera", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    
    ensure_directories(IMAGES_DIR, VIDEOS_DIR, EVENTS_DIR)
    cleanup_events()

    source = resolve_source(args)
    motion_gate = MotionGate()

    logger.info("Input source: %s", source.value)

    if source == FrameSource.IMAGE:
        process_images(motion_gate)
    elif source == FrameSource.VIDEO:
        process_videos(motion_gate)
    else:
        process_camera(motion_gate)

       

if __name__ == "__main__":
    main()
