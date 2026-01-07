"""
Main pipeline for Hornet Radar.

Responsibilities:
- Parse CLI arguments
- Select input source (camera / video / image)
- Feed frames into MotionGate
- Handle DetectionEvents (save + upload)

MotionGate is treated as a black box:
frame → (event | None, debug)
"""

import argparse
import os
import cv2

from config import *
from detection import load_model
from helpers import ensure_directories
from camera import Camera
from motion_gate import MotionGate
from event_storage import save_event, upload_event
from sources import FrameSource


# ============================================================
# Argument parsing
# ============================================================

parser = argparse.ArgumentParser(description="Hornet Radar main pipeline")

parser.add_argument('-v', '--videos', default=False, action='store_true', help="Analyses any .mp4 from detections/frames")
parser.add_argument('-i', '--images', default=False, action='store_true', help="Analyses any .jpg from detections/frames")

args = parser.parse_args()

def resolve_source() -> FrameSource:
    """
    Resolves input source based on CLI arguments.
    Priority:
    - images
    - videos
    - camera (default)
    """
    if args.images:
        return FrameSource.IMAGE
    if args.videos:
        return FrameSource.VIDEO
    return FrameSource.CAMERA


# ============================================================
# Image processing
# ============================================================

def process_images(motion_gate: MotionGate):
    print("Processing images...")

    for filename in os.listdir(IMAGES_DIR):
        if not filename.lower().endswith(".jpg"):
            continue

        path = os.path.join(IMAGES_DIR, filename)
        frame = cv2.imread(path)

        if frame is None:
            continue

        event, debug = motion_gate.process_frame(frame, FrameSource.IMAGE)

        if event and event.confidence >= CONFIDENCE_THRESHOLD:
            print(event)
            save_event(event, frame)
            upload_event(event)



# ============================================================
# Video processing
# ============================================================

def process_videos(motion_gate: MotionGate):
    print("Processing videos...")

    for filename in os.listdir(VIDEOS_DIR):
        if not filename.lower().endswith(".mp4"):
            continue

        path = os.path.join(VIDEOS_DIR, filename)
        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            print(f"Could not open video: {filename}")
            continue

        print(f"Processing video: {filename}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            event, debug = motion_gate.process_frame(frame, FrameSource.VIDEO)

            if event and event.confidence >= CONFIDENCE_THRESHOLD:
                save_event(event, frame)
                upload_event(event)
                break  # stop after first confirmed event

        cap.release()


# ============================================================
# Camera processing (live)
# ============================================================

def process_camera(motion_gate: MotionGate):
    print(f"Capturing from camera @ {CAMERA_FPS} FPS")

    cam = Camera()

    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue

            # Picamera returns RGB → OpenCV expects BGR
            if CAMERA_TYPE == "picamera2":
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            event, debug = motion_gate.process_frame(frame, FrameSource.CAMERA)

            if event and event.confidence >= CONFIDENCE_THRESHOLD:
                save_event(event, frame)
                upload_event(event)

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


# ============================================================
# Debug overlay
# ============================================================

def draw_debug_overlay(frame, debug: dict):
    """
    Renders debug information returned by MotionGate.
    """
    y = 20
    step = 22

    def line(text, color=(255, 255, 255)):
        nonlocal y
        cv2.putText(
            frame, text, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )
        y += step
    line(f"Press ESC to exit")
    line(f"Source: {debug.get('source')}")
    line(f"FPS: {debug.get('fps', 0):.1f}" if debug.get("fps") else "FPS: -")
    line(f"Motion: {'YES' if debug.get('motion') else 'NO'}",
         (0, 0, 255) if debug.get("motion") else (0, 255, 0))
    line(f"Tracking: {'YES' if debug.get('tracking') else 'NO'}",
         (0, 255, 255) if debug.get("tracking") else (150, 150, 150))
    line(f"Frames tracked: {debug.get('frames_tracked', 0)}")
    line(f"YOLO ran: {'YES' if debug.get('yolo_ran') else 'NO'}")


# ============================================================
# Main entry point
# ============================================================

def main():

    
    ensure_directories(IMAGES_DIR, LABELED_IMAGES_DIR, LABELED_IMAGES_THUMBS_DIR, VIDEOS_DIR, LABELED_VIDEOS_DIR, LABELED_VIDEOS_THUMBS_DIR)
    source = resolve_source()
    motion_gate = MotionGate()

    print(f"Input source: {source.value}")

    if source == FrameSource.IMAGE:
        process_images(motion_gate)

    elif source == FrameSource.VIDEO:
        process_videos(motion_gate)

    else:
        process_camera(motion_gate)

       

if __name__ == "__main__":
    main()
