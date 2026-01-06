# motion_gate.py
import cv2
from config import *
from detection import run_detection
from tracking_state import TrackingState
from event import DetectionEvent

# ============================================================
# Classes
# ============================================================

class MotionGate:
    """
    Handles motion detection, tracking and ROI-based YOLO detection.
    Produces DetectionEvent objects.
    """

    def __init__(self, model):
        self.model = model
        self.tracking_state = TrackingState()

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=MOTION_HISTORY,
            varThreshold=MOTION_VAR_THRESHOLD,
            detectShadows=False
        )

        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (MOTION_KERNEL_SIZE, MOTION_KERNEL_SIZE)
        )

        self.frame_count = 0

    def process_frame(self, frame):
    
        """
        Process one frame.
        Returns DetectionEvent or None.
        """
        event = None
        debug = {
            "motion": False,
            "tracking": self.tracking_state.active,
            "frames": self.tracking_state.frames_tracked,
            "yolo_done": self.tracking_state.detection_done,
            "bbox": self.tracking_state.bbox
        }

        self.frame_count += 1
        process_this_frame = (self.frame_count % FRAME_SKIP == 0)

        motion_detected = False
        motion_boxes = []

        # ------------------------------------------------
        # Motion Detection
        # ------------------------------------------------

        if process_this_frame:
            fg_mask = self.bg_subtractor.apply(frame)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)

            contours, _ = cv2.findContours(
                fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for c in contours:
                if cv2.contourArea(c) < MOTION_MIN_AREA:
                    continue

                x, y, w, h = cv2.boundingRect(c)
                motion_boxes.append((x, y, w, h))
                motion_detected = True
                debug["motion"] = motion_detected

            if motion_detected and not self.tracking_state.active:
                bbox = max(motion_boxes, key=lambda b: b[2] * b[3])
                tracker = create_tracker()
                tracker.init(frame, bbox)
                self.tracking_state.start(tracker, bbox)

            if self.tracking_state.active:
                ok, bbox = self.tracking_state.tracker.update(frame)
                if ok:
                    self.tracking_state.update(bbox)
                else:
                    self.tracking_state.reset()
                    return None, debug
                
            debug["tracking"] = self.tracking_state.active
            debug["frames"] = self.tracking_state.frames_tracked

            if self.tracking_state.is_stable(TRACKING_STABLE_FRAMES) and not self.tracking_state.detection_done:

                detections = run_yolo_on_roi(frame, self.tracking_state.bbox, self.model)

                self.tracking_state.detection_done = True
                debug["yolo_done"] = self.tracking_state.detection_done
                debug["bbox"] = self.tracking_state.bbox

                if not detections:
                    return None, debug

                event = DetectionEvent(
                    pi_id=PI_ID,
                    detections=detections,
                    model_name="yolov5",
                    tracking_bbox=self.tracking_state.bbox,
                    roi_bbox=self.tracking_state.bbox,
                    tracking_frames=self.tracking_state.frames_tracked,
                    frame_shape=frame.shape[:2]
                )


                
                return event, debug


            
            
            if self.tracking_state.is_timed_out(TRACKER_TIMEOUT):
                self.tracking_state.reset()

            return None, debug
        
        return event, debug  
# ============================================================
# Tracker Factory
# ============================================================

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
