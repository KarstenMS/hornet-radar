# motion_gate.py
import cv2
import time
from config import *
from detection import load_model, run_detection
from tracking_state import TrackingState
from event import DetectionEvent
from sources import FrameSource
from typing import Tuple, Optional, Dict


# ============================================================
# Classes
# ============================================================

class MotionGate:
    """
    MotionGate is responsible for:
    - motion detection
    - object tracking
    - triggering YOLO
    - creating DetectionEvent objects

    It does NOT:
    - save files
    - upload data
    - know argparse or CLI flags
    """

    def __init__(self):
        # --- YOLO ---
        self.model = load_model()

         # --- Tracking ---
        self.tracking_state = TrackingState()
        self.tracker = None

        # --- Motion detection ---
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=MOTION_HISTORY,
            varThreshold=MOTION_VAR_THRESHOLD,
            detectShadows=False
        )

        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (MOTION_KERNEL_SIZE, MOTION_KERNEL_SIZE)
        )

        # --- FPS (camera only) ---
        self.last_time = time.time()
        self.fps = 0.0

    def _process_camera(self, frame, debug):
        # --- FPS ---
        now = time.time()
        dt = now - self.last_time
        if dt > 0:
            self.fps = 1.0 / dt
        self.last_time = now
        debug["fps"] = self.fps

        motion_boxes = self._update_motion(frame, debug)
        self._update_tracking(frame, motion_boxes, debug)

        event = self._maybe_run_yolo(frame, debug)
        return event, debug

    def _process_video(self, frame, debug):
        motion_boxes = self._update_motion(frame, debug)
        self._update_tracking(frame, motion_boxes, debug)

        event = self._maybe_run_yolo(frame, debug)
        return event, debug

    def _process_image(self, frame, debug):
        debug["motion"] = False
        debug["tracking"] = False

        detections = run_detection(frame, self.model)
        if not detections:
            return None, debug

        event = DetectionEvent(
            pi_id=PI_ID,
            detections=detections,
            model_name=MODEL_NAME,
            frame_shape=frame.shape[:2],
        )

        debug["yolo_ran"] = True
        return event, debug

    def _update_motion(self, frame, debug):
        fg = self.bg_subtractor.apply(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.kernel)

        contours, _ = cv2.findContours(
            fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        boxes = []
        for c in contours:
            if cv2.contourArea(c) < MOTION_MIN_AREA:
                continue
            boxes.append(cv2.boundingRect(c))

        debug["motion"] = bool(boxes)
        return boxes

    def _update_tracking(self, frame, motion_boxes, debug):
        if motion_boxes and not self.tracking_state.active:
            bbox = max(motion_boxes, key=lambda b: b[2] * b[3])
            self.tracker = self._create_tracker()
            self.tracker.init(frame, bbox)
            self.tracking_state.start(self.tracker, bbox)

        if not self.tracking_state.active:
            return

        ok, bbox = self.tracking_state.tracker.update(frame)
        if ok:
            self.tracking_state.update(bbox)
            debug["tracking"] = True
            debug["frames_tracked"] = self.tracking_state.frames_tracked
        else:
            self.tracking_state.reset()

        if self.tracking_state.is_timed_out(TRACKER_TIMEOUT):
            self.tracking_state.reset()

    def _maybe_run_yolo(self, frame, debug):

        if not self.tracking_state.is_stable(TRACKING_STABLE_FRAMES):
            return None

        if self.tracking_state.detection_done:
            return None

        roi, _ = extract_roi(frame, self.tracking_state.bbox)
        detections = run_detection(roi, self.model)

        self.tracking_state.detection_done = True
        debug["yolo_ran"] = True

        if not detections:
            return None

        return DetectionEvent(
            pi_id=PI_ID,
            detections=detections,
            model_name=MODEL_NAME,
            tracking_bbox=self.tracking_state.bbox,
            tracking_frames=self.tracking_state.frames_tracked,
            frame_shape=frame.shape[:2],
        )      

    def create_tracker():

        t = TRACKER_TYPE.upper()

        if t == "KCF" and hasattr(cv2, "TrackerKCF_create"):
            print("Tracker: KCF")
            return cv2.TrackerKCF_create()

        if t == "CSRT" and hasattr(cv2, "TrackerCSRT_create"):
            print("Tracker: CSRT")
            return cv2.TrackerCSRT_create()

        if t == "MOSSE" and hasattr(cv2, "TrackerMOSSE_create"):
            print("Tracker: MOSSE")
            return cv2.TrackerMOSSE_create()

        if t == "AUTO":
            if hasattr(cv2, "TrackerKCF_create"):
                print("Tracker: AUTO → KCF")
                return cv2.TrackerKCF_create()
            if hasattr(cv2, "TrackerCSRT_create"):
                print("Tracker: AUTO → CSRT")
                return cv2.TrackerCSRT_create()

        raise RuntimeError("No suitable OpenCV tracker available")

# ============================================================
# Tracking helpers
# ============================================================



def draw_tracking(frame, bbox):
    x, y, w, h = map(int, bbox)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.putText(frame, "TRACKING",
                (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2)

def extract_roi(frame, bbox):
    x, y, w, h = map(int, bbox)
    h_f, w_f = frame.shape[:2]

    x = max(0, x)
    y = max(0, y)
    w = min(w, w_f - x)
    h = min(h, h_f - y)

    roi = frame[y:y + h, x:x + w]
    return roi, (x, y)

def run_yolo_on_roi(frame, bbox, model):
    roi, offset = extract_roi(frame, bbox)
    if roi.size == 0:
        return []

    predictions = run_detection(roi, model)

    detections = []
    for p in predictions:
        x1, y1, x2, y2 = p["bbox"]
        detections.append({
            "class_id": p["class_id"],
            "confidence": p["confidence"],
            "bbox": [
                x1 + offset[0],
                y1 + offset[1],
                x2 + offset[0],
                y2 + offset[1],
            ]
        })

    return detections

    if not self.tracking_state.is_stable(TRACKING_STABLE_FRAMES):
        return None

    if self.tracking_state.detection_done:
        return None

    roi, _ = extract_roi(frame, self.tracking_state.bbox)
    detections = run_detection(roi, self.model)

    self.tracking_state.detection_done = True
    debug["yolo_ran"] = True

    if not detections:
        return None

    return DetectionEvent(
        pi_id=PI_ID,
        detections=detections,
        model_name="yolo-motion",
        tracking_bbox=self.tracking_state.bbox,
        tracking_frames=self.tracking_state.frames_tracked,
        frame_shape=frame.shape[:2],
    )

def process_frame(self, frame, source: FrameSource) -> Tuple[Optional[DetectionEvent], Dict]:

    """
    Process one frame.
    Returns DetectionEvent or None.
    """
    debug = {
        "source": source.value,
        "motion": False,
        "tracking": False,
        "frames": 0,
        "yolo_done": False,
        "fps": None,
    }

    if source == FrameSource.IMAGE:
        return self._process_image(frame, debug), debug

    if source == FrameSource.VIDEO:
        return self._process_video(frame, debug), debug

    if source == FrameSource.CAMERA:
        return self._process_camera(frame, debug), debug
    
    raise ValueError(f"Unsupported FrameSource: {source}")        