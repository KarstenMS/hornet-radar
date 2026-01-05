"""
motion_gate.py

Handles motion detection, object tracking and gated YOLO inference.
This module is responsible for deciding *when* an object is relevant,
not for storing or uploading results.

Pipeline:
Camera frame
→ Motion detection (MOG2)
→ Start tracker on largest motion
→ Track object
→ After stable tracking → run YOLO once on ROI
"""

import cv2
import time
from typing import Optional, Tuple, List
from camera import Camera
from config import *
from detection import load_model, run_detection


# ============================================================
# Tracking State
# ============================================================

class TrackingState:
    """
    Encapsulates the full lifecycle of one tracking event.
    """

    def __init__(self):
        self.active: bool = False
        self.tracker = None

        self.bbox: Optional[Tuple[int, int, int, int]] = None

        self.frames_tracked: int = 0
        self.detection_done: bool = False

        self.start_time: float = 0.0
        self.last_update: float = 0.0

    # ----------------------------
    # Lifecycle
    # ----------------------------
    def start(self, tracker, bbox):
        self.tracker = tracker
        self.bbox = bbox
        self.active = True

        self.frames_tracked = 0
        self.detection_done = False

        now = time.time()
        self.start_time = now
        self.last_update = now

    def update(self, bbox):
        self.bbox = bbox
        self.frames_tracked += 1
        self.last_update = time.time()

    def reset(self):
        self.active = False
        self.tracker = None
        self.bbox = None
        self.frames_tracked = 0
        self.detection_done = False
        self.start_time = 0.0
        self.last_update = 0.0

    # ----------------------------
    # Helper
    # ----------------------------
    def is_stable(self, min_frames: int) -> bool:
        return self.active and self.frames_tracked >= min_frames

    def is_timed_out(self, timeout_s: float) -> bool:
        return self.active and (time.time() - self.last_update) > timeout_s


# ============================================================
# Tracker Factory
# ============================================================

def create_tracker():
    """
    Create OpenCV tracker based on config.
    """
    if TRACKER_TYPE == "KCF":
        return cv2.TrackerKCF_create()
    elif TRACKER_TYPE == "CSRT":
        return cv2.TrackerCSRT_create()
    else:
        # AUTO fallback
        try:
            return cv2.TrackerKCF_create()
        except Exception:
            return cv2.TrackerCSRT_create()


# ============================================================
# Drawing helpers
# ============================================================

def draw_tracking(frame, bbox):
    """
    Draw tracking bounding box.
    """
    x, y, w, h = map(int, bbox)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.putText(
        frame,
        "TRACKING",
        (x, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        2,
    )


def draw_motion_boxes(frame, boxes):
    """
    Draw motion bounding boxes.
    """
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 1)


# ============================================================
# Motion Gate
# ============================================================

def motion_gate_loop(camera):
    """
    Main motion gate loop.

    Returns:
        Generator yielding YOLO detections when available.
    """

    # --- Background subtractor ---
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=MOTION_HISTORY,
        varThreshold=MOTION_VAR_THRESHOLD,
        detectShadows=False,
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (MOTION_KERNEL_SIZE, MOTION_KERNEL_SIZE),
    )

    # --- State ---
    state = TrackingState()
    yolo_model = load_model()

    frame_count = 0
    last_time = time.time()
    fps = 0.0

    print("Motion-Gate started (ESC to quit)")

    # ========================================================
    # Main Loop
    # ========================================================
    while True:
        frame = camera.get_frame()
        if frame is None:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        display = frame.copy()

        # --- FPS ---
        now = time.time()
        dt = now - last_time
        if dt > 0:
            fps = 1.0 / dt
        last_time = now

        frame_count += 1
        process_frame = (frame_count % FRAME_SKIP == 0)

        motion_detected = False
        motion_boxes: List[Tuple[int, int, int, int]] = []

        # ====================================================
        # Motion Detection
        # ====================================================
        if process_frame:
            fg_mask = bg_subtractor.apply(frame)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(
                fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for c in contours:
                if cv2.contourArea(c) < MOTION_MIN_AREA:
                    continue

                x, y, w, h = cv2.boundingRect(c)
                motion_boxes.append((x, y, w, h))
                motion_detected = True

            draw_motion_boxes(display, motion_boxes)

            # ------------------------------------------------
            # Start tracking on largest motion
            # ------------------------------------------------
            if motion_detected and not state.active:
                x, y, w, h = max(
                    motion_boxes, key=lambda b: b[2] * b[3]
                )
                tracker = create_tracker()
                tracker.init(frame, (x, y, w, h))
                state.start(tracker, (x, y, w, h))

        # ====================================================
        # Tracking
        # ====================================================
        if state.active:
            ok, bbox = state.tracker.update(frame)
            if ok:
                state.update(bbox)
                draw_tracking(display, bbox)
            else:
                state.reset()

        # ====================================================
        # YOLO (run once per stable tracking)
        # ====================================================
        if state.is_stable(TRACKING_STABLE_FRAMES) and not state.detection_done:
            detections = run_detection(frame, state.bbox, yolo_model)
            state.detection_done = True

            if detections:
                yield detections, state.bbox

        # ====================================================
        # Timeout
        # ====================================================
        if state.is_timed_out(TRACKER_TIMEOUT):
            state.reset()

        # ====================================================
        # Debug Overlay
        # ====================================================
        if SHOW_DEBUG_VIDEO:
            y = 20
            step = 22

            cv2.putText(display, f"PI: {PI_ID}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            y += step
            cv2.putText(display, f"FPS: {fps:.1f}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            y += step
            cv2.putText(display,
                        f"Motion: {'YES' if motion_detected else 'NO'}",
                        (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,0,255) if motion_detected else (0,255,0),
                        2)
            y += step
            cv2.putText(display,
                        f"Tracking: {'YES' if state.active else 'NO'}",
                        (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,255,255) if state.active else (150,150,150),
                        2)
            y += step
            cv2.putText(display,
                        f"Frames tracked: {state.frames_tracked}",
                        (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255,255,255),
                        2)
            y += step
            cv2.putText(display,
                        f"YOLO done: {'YES' if state.detection_done else 'NO'}",
                        (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,255,0) if state.detection_done else (0,0,255),
                        2)

            cv2.imshow("Hornet Debug", display)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cv2.destroyAllWindows()
    print("Motion-Gate stopped")
