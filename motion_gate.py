# motion_gate.py
import cv2
import time
from config import *
from detection import load_model, run_detection
from tracking_state import TrackingState
from event import DetectionEvent
from sources import FrameSource
from typing import Tuple, Optional, Dict
from motion_vectors import vector_from_points

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
        self.frame_count = 0

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
            return self._process_image(frame, debug)

        if source == FrameSource.VIDEO:
            return self._process_video(frame, debug)

        if source == FrameSource.CAMERA:
            return self._process_camera(frame, debug)
        
        raise ValueError(f"Unsupported FrameSource: {source}")    

    def _process_camera(self, frame, debug):
        event = None  
        self.source = "Camera"
        # --- FPS ---
        now = time.time()
        dt = now - self.last_time
        if dt > 0:
            self.fps = 1.0 / dt
        self.last_time = now
        debug["fps"] = self.fps

        self.frame_count += 1
        process_this_frame = (self.frame_count % FRAME_SKIP == 0)

        if process_this_frame:
            motion_boxes = self._update_motion(frame, debug)
            event = self._update_tracking(frame, motion_boxes, debug)
            if event:
                return event, debug
        else:
            if self.tracking_state.active:
                ok, bbox = self.tracking_state.tracker.update(frame)
                if ok:
                    self.tracking_state.update(bbox)
                    debug["tracking"] = True

        self._maybe_run_yolo(frame, debug)
        return None, debug



    def _process_video(self, frame, debug):
        self.source = "Video"
        debug["motion"] = False
        debug["tracking"] = False

        detections = None
        self.frame_count += 1
        process_this_frame = (self.frame_count % FRAME_SKIP == 0)

        if process_this_frame:
            detections = run_detection(frame, self.model)
        if not detections:
            return None, debug
        
        event = DetectionEvent(
            pi_id=PI_ID,
            detections=detections,
            tracking_bbox=None,
            tracking_frames=0,
            frame=frame,
            model_name=MODEL_NAME,
            source=self.source,
            frame_shape=frame.shape[:2],
        )

        debug["yolo_ran"] = True
        debug["yolo_done"] = True
        return event, debug

    def _process_image(self, frame, debug):
        self.source = "Image"
        debug["motion"] = False
        debug["tracking"] = False
   
        detections = run_detection(frame, self.model)
        if not detections:
            return None, debug

        event = DetectionEvent(
            pi_id=PI_ID,
            detections=detections,
            tracking_bbox=None,
            tracking_frames=0,
            model_name=MODEL_NAME,
            source=self.source,
            frame_shape=frame.shape[:2],
        )

        debug["yolo_ran"] = True
        return event, debug

    def _update_motion(self, frame, debug):
        #print("Update Motion run")
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
        #print("Update Tracking run")

        # -------------------------------------------------
        # 1. Tracker starten
        # -------------------------------------------------
        if motion_boxes and not self.tracking_state.active:
            bbox = max(motion_boxes, key=lambda b: b[2] * b[3])
            self.tracker = self._create_tracker()
            self.tracker.init(frame, bbox)
            print("TRACKER INIT BBOX:", bbox)
            print("FRAME SHAPE:", frame.shape)

            self.tracking_state.start(self.tracker, bbox, frame.shape[:2])
            return None

        if not self.tracking_state.active:
            print("No active tracker")
            return None

        # -------------------------------------------------
        # 2. Tracker updaten
        # -------------------------------------------------
        ok, bbox = self.tracking_state.tracker.update(frame)



        if ok:
            # 🔍 Geometrie prüfen
            plausible = self.bbox_is_plausible(bbox, frame.shape[:2])

            if not plausible:
                print("Tracker geometry invalid")

                if self.tracking_state.confirmed:
                    event = self._finalize_event()
                    self.tracking_state.reset()
                    return event

                self.tracking_state.reset()
                return None

            # ✅ alles okay
            self.tracking_state.update(bbox)

            if not self.tracking_state.confirmed:
                self.tracking_state.last_good_frame = frame.copy()
                self.tracking_state.last_good_frame_shape = frame.shape
            else:
                self.tracking_state.frames_since_confirmed += 1

            debug["tracking"] = True
            debug["frames_tracked"] = self.tracking_state.frames_tracked
            debug["tracking_bbox"] = tuple(map(int, bbox))

            return None

        # -------------------------------------------------
        # 3. Tracker verloren → Event ggf. finalisieren
        # -------------------------------------------------
        print("Tracker lost")

        if self.tracking_state.confirmed:
            event = self._finalize_event()
            self.tracking_state.reset()
            return event

        self.tracking_state.reset()
        return None


    def _maybe_run_yolo(self, frame, debug):
        #print(f"Maybe Run Yolo")

        # --- nur nach stabiler Trackingphase ---
        if not self.tracking_state.is_stable(TRACKING_STABLE_FRAMES):
            return None
        
         # --- YOLO nur einmal ---
        if self.tracking_state.detection_done:
            return None
        
        # --- ROI aus aktueller Tracking-BBox ---
        if self.tracking_state.bbox is None:
            return None
    
        roi, offset, roi_box = self._extract_roi(frame, self.tracking_state.bbox)
        debug["roi_bbox"] = roi_box
        if roi.size == 0:
            debug["roi_bbox"] = None
            return None
        
        detections = run_detection(roi, self.model)   
        print(f"Detection done, found {len(detections)} objects")
        debug["yolo_ran"] = True 

        if not detections:
            return None
       
        # --- YOLO BBox → Frame-Koordinaten ---
        frame_detections = []
        print("Detections:", detections)
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            frame_detections.append({
                **d,
                "bbox": (
                int(x1 + offset[0]),
                int(y1 + offset[1]),
                int(x2 + offset[0]),
                int(y2 + offset[1]),
                )
            })

        best_det = frame_detections[0]   
        print(f"Hornet detected!!!")

        self.tracking_state.confirmed = True
        self.tracking_state.detection_done = True

        # ✅ DAS Frame, auf dem YOLO lief
        self.tracking_state.confirmed_frame = frame
        self.tracking_state.confirmed_frame_shape = frame.shape

        # ✅ DIE YOLO BOX (nicht Tracker!)
        self.tracking_state.confirmed_bbox = best_det["bbox"]
        self.tracking_state.confirmed_yolo_bbox = best_det["bbox"] 

        # ✅ Bewegung weiter sammeln
        self.tracking_state.confirmed_centers = list(self.tracking_state.centers)
        self.tracking_state.detections = frame_detections
        
        debug["yolo_bbox"] = best_det["bbox"]

        print("YOLO CONFIRMED BBOX:", best_det["bbox"])
        print("TRACKER BBOX (live):", self.tracking_state.bbox)

        return None


    def _finalize_event(self) -> DetectionEvent:
        print("Finalize Event")
        

        # --- Extract data from tracking state ---
        centers = self.tracking_state.confirmed_centers
  
        # --- Compute movement vectors ---
        approach_vec = vector_from_points(centers, mode="approach")
        departure_vec = vector_from_points(centers, mode="departure")


        return DetectionEvent(
            pi_id=PI_ID,
            detections=self.tracking_state.detections,
            model_name=MODEL_NAME,
            source=self.source,
            frame=self.tracking_state.confirmed_frame,
            tracking_bbox=self.tracking_state.confirmed_yolo_bbox,
            tracking_frames=len(centers),
            frame_shape=self.tracking_state.confirmed_frame_shape,
            approach_vec=approach_vec,
            departure_vec=departure_vec,
            dwell_time=self.tracking_state.dwell_time,
        )


    def _create_tracker(self):

        t = TRACKER_TYPE.upper()

        if t == "KCF" and hasattr(cv2, "TrackerKCF_create"):
            print("Tracking started using: KCF")
            return cv2.TrackerKCF_create()

        if t == "CSRT" and hasattr(cv2, "TrackerCSRT_create"):
            print("Tracking started using: CSRT")
            return cv2.TrackerCSRT_create()

        if t == "MOSSE" and hasattr(cv2, "TrackerMOSSE_create"):
            print("Tracking started using: MOSSE")
            return cv2.TrackerMOSSE_create()

        if t == "AUTO":
            if hasattr(cv2, "TrackerKCF_create"):
                print("Tracking started using: AUTO → KCF")
                return cv2.TrackerKCF_create()
            if hasattr(cv2, "TrackerCSRT_create"):
                print("Tracking started using: AUTO → CSRT")
                return cv2.TrackerCSRT_create()

        raise RuntimeError("No suitable OpenCV tracker available")
    
    def _extract_roi(self, frame, bbox):
        x, y, w, h = map(int, bbox)
        h_f, w_f = frame.shape[:2]

        x = max(0, x)
        y = max(0, y)
        w = min(w, w_f - x)
        h = min(h, h_f - y)
        roi = frame[y:y + h, x:x + w]
        roi_box = (x, y, x + w, y + h)
        return roi, (x, y), roi_box

    def xywh_to_xyxy(self,bbox):
        x, y, w, h = bbox
        return (x, y, x + w, y + h)
    
    def bbox_is_plausible(self, bbox, frame_shape):
        x, y, w, h = bbox
        fh, fw = frame_shape

        area = w * h
        frame_area = fw * fh
        area_ratio = area / frame_area

        aspect = w / h if h > 0 else 0

        # --- Area checks ---
        if area_ratio > TRACKER_MAX_AREA_RATIO:
            return False

        if area_ratio < TRACKER_MIN_AREA_RATIO:
            return False

        # --- Aspect ratio ---
        if aspect > TRACKER_MAX_ASPECT_RATIO or aspect < TRACKER_MIN_ASPECT_RATIO:
            return False

        # --- Edge hugging ---
        margin_x = fw * TRACKER_EDGE_MARGIN_RATIO
        margin_y = fh * TRACKER_EDGE_MARGIN_RATIO

        if (
            x <= margin_x or
            y <= margin_y or
            x + w >= fw - margin_x or
            y + h >= fh - margin_y
        ):
            return False

        return True
