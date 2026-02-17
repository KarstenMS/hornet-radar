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

        # --- Frame count & skipping ---
        self.frame_count += 1
        process_this_frame = (self.frame_count % FRAME_SKIP == 0)

        # --- Motion Detection (only every N Frames) ---
        motion_boxes = []
        if process_this_frame:
            motion_boxes = self._update_motion(frame, debug)
        else:
            debug["motion"] = False

        # --- Tracking (always)
        event = self._update_tracking(frame, motion_boxes, debug)      
        if event:
            return event, debug
        
        # -- Yolo (if tracking, stable, and not done yet)
        self._maybe_run_yolo(frame, debug)

        return None, debug



    def _process_video(self, frame, debug):
        self.source = "Video"
        debug["motion"] = False
        debug["tracking"] = False
        #debug["confirmed"] = False

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
        debug["motion_boxes"] = boxes
        return boxes

    def _update_tracking(self, frame, motion_boxes, debug):
        #print("Update Tracking run")

        # -------------------------------------------------
        # 1. Tracker starten
        # -------------------------------------------------
        if motion_boxes and not self.tracking_state.active:
            bbox = max(motion_boxes, key=lambda b: b[2] * b[3])

            x, y, w, h = bbox
            fh, fw = frame.shape[:2]

            area_ratio = (w * h) / (fw * fh)

            if area_ratio > TRACKER_INIT_MAX_AREA_RATIO:
                print("Rejecting tracker init: motion bbox too large")
                return None


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

                self.tracking_state.invalid_frames += 1

                if self.tracking_state.invalid_frames > TRACKER_MAX_INVALID_FRAMES:
                     print("Tracker lost > abort")

                if self.tracking_state.confirmed:
                    # Nur finalisieren, wenn wir genug Frames nach Confirmation hatten
                    if self.tracking_state.frames_since_confirmed >= MIN_POST_CONFIRM_FRAMES:
                        event = self._finalize_event()
                        self.tracking_state.reset()
                        return event
                    else:
                        # Noch nicht genug Daten gesammelt → ignorieren
                        return None

                self.tracking_state.reset()
                return None


            # ✅ alles okay
            self.tracking_state.invalid_frames = 0
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
        print("No Tracking")

        if self.tracking_state.confirmed:
            event = self._finalize_event()
            self.tracking_state.reset()
            return event

        self.tracking_state.reset()
        return None


    def _maybe_run_yolo(self, frame, debug):

        # --- YOLO nur einmal ---
        if self.tracking_state.detection_done:
            return None

        # --- nur nach stabiler Trackingphase ---
        if not self.tracking_state.is_stable(TRACKING_STABLE_FRAMES):
            return None
        
        if self.tracking_state.yolo_attempts >= MAX_YOLO_ATTEMPTS:
            print("YOLO attempts exhausted → abort tracking")
            self.tracking_state.reset()
            return None
        
        # --- YOLO auf FULL FRAME ---
        detections = run_detection(frame, self.model)

        print(f"Detection done, found {len(detections)} objects")
        debug["yolo_ran"] = True
        self.tracking_state.yolo_attempts += 1

        if not detections:
            return None
       
        best_det = detections[0]   
        print(f"Hornet detected!!!")

        # Tracker neu auf YOLO-Box setzen
        new_bbox = best_det["bbox"]
        x1, y1, x2, y2 = new_bbox
        w = x2 - x1
        h = y2 - y1

        self.tracking_state.tracker = self._create_tracker()
        self.tracking_state.tracker.init(frame, (x1, y1, w, h))
        self.tracking_state.bbox = (x1, y1, w, h)

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
        self.tracking_state.detections = detections

        # ✅ Labels und Confidence speichern
        CLASS_MAP = {
            0: "AH",
            1: "EH"
        }
        label = CLASS_MAP.get(best_det["class_id"], "?")
        conf = best_det["confidence"]

        debug["confirmed"] = True
        debug["confirmed_label"] = label
        debug["confirmed_conf"] = conf
        debug["yolo_bbox"] = best_det["bbox"]

        return None


    def _finalize_event(self) -> DetectionEvent:
        print("Finalize Event")
        

        # --- Extract data from tracking state ---
        centers = self.tracking_state.confirmed_centers
  
        # --- Compute movement vectors ---
        approach_vec = vector_from_points(centers, mode="approach")
        departure_vec = vector_from_points(centers, mode="departure")
        print(f"Approach vector: {approach_vec}, Departure vector: {departure_vec}")


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
            if hasattr(cv2, "TrackerCSRT_create"):
                print("Tracking started using: AUTO → CSRT")
                return cv2.TrackerCSRT_create()
            if hasattr(cv2, "TrackerKCF_create"):
                print("Tracking started using: AUTO → KCF")
                return cv2.TrackerKCF_create()

        raise RuntimeError("No suitable OpenCV tracker available")
    

    def xywh_to_xyxy(self,bbox):
        x, y, w, h = bbox
        return (x, y, x + w, y + h)
    
    def bbox_is_plausible(self, bbox, frame_shape):
        x, y, w, h = bbox
        fh, fw = frame_shape

        area_ratio = (w * h) / (fw * fh)
        aspect = w / h if h > 0 else 0


        # --- always Area checks ---
        if area_ratio > TRACKER_MAX_AREA_RATIO:
            return False

        # --- Aspect ratio after confirmation---
        if self.tracking_state.confirmed:
            if area_ratio < TRACKER_MIN_AREA_RATIO:
                return False

            if aspect > TRACKER_MAX_ASPECT_RATIO or aspect < TRACKER_MIN_ASPECT_RATIO:
                return False

            # --- Edge hugging ---
            margin_x = fw * TRACKER_EDGE_MARGIN_RATIO
            margin_y = fh * TRACKER_EDGE_MARGIN_RATIO


            if self.tracking_state.frames_tracked > 3:
                if(
                    x <= margin_x or
                    y <= margin_y or
                    x + w >= fw - margin_x or
                    y + h >= fh - margin_y
                ):
                    return False

        return True
