"""Hornet Radar: motion gating pipeline (background subtraction, tracking, YOLO trigger)."""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional, Tuple

import cv2

from config import (
    FRAME_SKIP,
    MAX_YOLO_ATTEMPTS,
    MIN_POST_CONFIRM_FRAMES,
    MODEL_NAME,
    MOTION_HISTORY,
    MOTION_KERNEL_SIZE,
    MOTION_MIN_AREA,
    MOTION_VAR_THRESHOLD,
    PI_ID,
    TRACKER_EDGE_MARGIN_RATIO,
    TRACKER_INIT_MAX_AREA_RATIO,
    TRACKER_MAX_AREA_RATIO,
    TRACKER_MAX_ASPECT_RATIO,
    TRACKER_MAX_INVALID_FRAMES,
    TRACKER_MIN_AREA_RATIO,
    TRACKER_MIN_ASPECT_RATIO,
    TRACKER_TYPE,
    TRACKING_STABLE_FRAMES,
)
from detection import load_model, run_detection
from event import DetectionEvent
from motion_vectors import vector_from_points
from sources import FrameSource
from tracking_state import TrackingState

logger = logging.getLogger(__name__)


class MotionGate:
    """Main stateful pipeline.

    Responsibilities:
        - motion detection (MOG2)
        - object tracking (OpenCV tracker)
        - trigger YOLO once tracking is stable
        - create <DetectionEvent> once tracking ends after confirmation

    The public API is <process_frame>.
    """

    def __init__(self) -> None:
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
            detectShadows=False,
        )
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (MOTION_KERNEL_SIZE, MOTION_KERNEL_SIZE)
        )

        # --- FPS (camera only) ---
        self.last_time = time.time()
        self.fps = 0.0

        # Human readable for the event
        self.source_name = "Unknown"

    def process_frame(self, frame, source: FrameSource) -> Tuple[Optional[DetectionEvent], Dict]:
        """Process one frame and return (event, debug).

        Args:
            frame: OpenCV frame (numpy array).
            source: Frame origin (camera, video, image).

        Returns:
            event: A <DetectionEvent> or None
            debug: A dict with debug values for overlays / logging
        """
        debug: Dict = {
            "source": source.value,
            "motion": False,
            "tracking": False,
            "frames_tracked": 0,
            "yolo_ran": False,
            "confirmed": False,
            "tracking_bbox": None,
            "motion_boxes": [],
            "fps": None,
            "bbox_plausible": None,
        }

        if source == FrameSource.IMAGE:
            self.source_name = "Image"
            return self._process_image(frame, debug)

        if source == FrameSource.VIDEO:
            self.source_name = "Video"
            return self._process_video(frame, debug)

        if source == FrameSource.CAMERA:
            self.source_name = "Camera"
            return self._process_camera(frame, debug)

        raise ValueError(f"Unsupported FrameSource: {source}")

    def _process_camera(self, frame, debug: Dict) -> Tuple[Optional[DetectionEvent], Dict]:
        """Camera mode: motion-gate + tracker + YOLO trigger."""
        # FPS
        now = time.time()
        dt = now - self.last_time
        if dt > 0:
            self.fps = 1.0 / dt
        self.last_time = now
        debug["fps"] = self.fps

        # Frame skipping for motion detection
        self.frame_count += 1
        process_this_frame = (self.frame_count % FRAME_SKIP == 0)

        motion_boxes = []
        if process_this_frame:
            motion_boxes = self._update_motion(frame, debug)

        event = self._update_tracking(frame, motion_boxes, debug)
        if event:
            return event, debug

        self._maybe_run_yolo(frame, debug)
        return None, debug

    def _process_video(self, frame, debug: Dict) -> Tuple[Optional[DetectionEvent], Dict]:
        """Video mode: run YOLO on every Nth frame (no tracking)."""
        self.frame_count += 1
        if self.frame_count % FRAME_SKIP != 0:
            return None, debug

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
            source=self.source_name,
            frame_shape=frame.shape[:2],
        )
        debug["yolo_ran"] = True
        return event, debug

    def _process_image(self, frame, debug: Dict) -> Tuple[Optional[DetectionEvent], Dict]:
        """Image mode: run YOLO once (no tracking)."""
        detections = run_detection(frame, self.model)
        if not detections:
            return None, debug

        event = DetectionEvent(
            pi_id=PI_ID,
            detections=detections,
            tracking_bbox=None,
            tracking_frames=0,
            model_name=MODEL_NAME,
            source=self.source_name,
            frame_shape=frame.shape[:2],
        )
        debug["yolo_ran"] = True
        return event, debug

    def _update_motion(self, frame, debug: Dict):
        """Update background model and return motion bounding boxes."""
        fg = self.bg_subtractor.apply(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.kernel)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            if cv2.contourArea(c) < MOTION_MIN_AREA:
                continue
            boxes.append(cv2.boundingRect(c))

        debug["motion"] = bool(boxes)
        debug["motion_boxes"] = boxes
        return boxes

    def _update_tracking(self, frame, motion_boxes, debug: Dict) -> Optional[DetectionEvent]:
        """Start/update tracker. Finalize confirmed events when tracking ends."""
        # 1) Start a new tracker on motion
        if motion_boxes and not self.tracking_state.active:
            bbox = max(motion_boxes, key=lambda b: b[2] * b[3])
            x, y, w, h = bbox
            fh, fw = frame.shape[:2]
            area_ratio = (w * h) / (fw * fh)
            if area_ratio > TRACKER_INIT_MAX_AREA_RATIO:
                logger.debug("Rejecting tracker init: motion bbox too large (ratio=%.3f)", area_ratio)
                return None

            self.tracker = self._create_tracker()
            self.tracker.init(frame, bbox)
            self.tracking_state.start(self.tracker, bbox, frame.shape[:2])
            return None

        # 2) No active tracker
        if not self.tracking_state.active:
            return None

        # 3) Update tracker
        ok, bbox = self.tracking_state.tracker.update(frame)
        if not ok:
            # Tracker lost
            return self._maybe_finalize_on_loss()

        plausible = self._bbox_is_plausible(bbox, frame.shape[:2])
        debug["bbox_plausible"] = plausible

        if not plausible:
            self.tracking_state.invalid_frames += 1
            if self.tracking_state.invalid_frames > TRACKER_MAX_INVALID_FRAMES:
                return self._maybe_finalize_on_loss(min_post_confirm=True)
            return None

        # Geometry OK
        self.tracking_state.invalid_frames = 0
        self.tracking_state.update(bbox)

        if not self.tracking_state.confirmed:
            self.tracking_state.last_good_frame = frame.copy()
            self.tracking_state.last_good_frame_shape = frame.shape
        else:
            self.tracking_state.frames_since_confirmed += 1

        # Debug output
        debug["tracking"] = True
        debug["frames_tracked"] = self.tracking_state.frames_tracked
        debug["tracking_bbox"] = tuple(map(int, bbox))
        debug["confirmed"] = self.tracking_state.confirmed
        if self.tracking_state.confirmed:
            debug["confirmed_label"] = self.tracking_state.confirmed_label
            debug["confirmed_conf"] = self.tracking_state.confirmed_confidence

        return None

    def _maybe_finalize_on_loss(self, *, min_post_confirm: bool = False) -> Optional[DetectionEvent]:
        """Finalize an event when tracking is lost, if confirmed."""
        if self.tracking_state.confirmed:
            if min_post_confirm and self.tracking_state.frames_since_confirmed < MIN_POST_CONFIRM_FRAMES:
                return None
            event = self._finalize_event()
            self.tracking_state.reset()
            return event

        self.tracking_state.reset()
        return None

    def _maybe_run_yolo(self, frame, debug: Dict) -> None:
        """Run YOLO once after stable tracking and store the confirmation in state."""
        if self.tracking_state.detection_done:
            return

        if self.tracking_state.frames_tracked < TRACKING_STABLE_FRAMES:
            return

        if self.tracking_state.yolo_attempts >= MAX_YOLO_ATTEMPTS:
            logger.debug("YOLO attempts exhausted -> abort tracking")
            self.tracking_state.reset()
            return

        detections = run_detection(frame, self.model)
        debug["yolo_ran"] = True
        self.tracking_state.yolo_attempts += 1

        if not detections:
            return

        # Pick the highest-confidence detection
        best_det = max(detections, key=lambda d: float(d.get("confidence", 0.0)))
        label = "AH" if best_det.get("class_id") == 1 else "EH"
        conf = float(best_det.get("confidence", 0.0))

        # Re-initialize tracker on the YOLO bbox
        x1, y1, x2, y2 = best_det["bbox"]
        w, h = x2 - x1, y2 - y1

        self.tracking_state.tracker = self._create_tracker()
        self.tracking_state.tracker.init(frame, (x1, y1, w, h))
        self.tracking_state.bbox = (x1, y1, w, h)

        self.tracking_state.confirmed = True
        self.tracking_state.detection_done = True
        self.tracking_state.confirmed_label = label
        self.tracking_state.confirmed_confidence = conf
        self.tracking_state.confirmed_frame = frame
        self.tracking_state.confirmed_frame_shape = frame.shape
        self.tracking_state.confirmed_yolo_bbox = best_det["bbox"]
        self.tracking_state.detections = detections

    def _finalize_event(self) -> DetectionEvent:
        """Create a <DetectionEvent> from current tracking state."""
        centers = self.tracking_state.centers
        approach_vec = vector_from_points(centers, mode="approach")
        departure_vec = vector_from_points(centers, mode="departure")

        return DetectionEvent(
            pi_id=PI_ID,
            detections=self.tracking_state.detections,
            model_name=MODEL_NAME,
            source=self.source_name,
            frame=self.tracking_state.confirmed_frame,
            tracking_bbox=self.tracking_state.confirmed_yolo_bbox,
            tracking_frames=len(centers),
            frame_shape=(self.tracking_state.confirmed_frame_shape[:2] if self.tracking_state.confirmed_frame_shape is not None else None),
            approach_vec=approach_vec,
            departure_vec=departure_vec,
            dwell_time=self.tracking_state.dwell_time,
        )

    def _create_tracker(self):
        """Create an OpenCV tracker instance based on configuration."""
        t = TRACKER_TYPE.upper()

        if t == "KCF" and hasattr(cv2, "TrackerKCF_create"):
            return cv2.TrackerKCF_create()
        if t == "CSRT" and hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
        if t == "MOSSE" and hasattr(cv2, "TrackerMOSSE_create"):
            return cv2.TrackerMOSSE_create()

        if t == "AUTO":
            if hasattr(cv2, "TrackerCSRT_create"):
                return cv2.TrackerCSRT_create()
            if hasattr(cv2, "TrackerKCF_create"):
                return cv2.TrackerKCF_create()

        raise RuntimeError("No suitable OpenCV tracker available")

    def _bbox_is_plausible(self, bbox, frame_shape) -> bool:
        """Apply basic plausibility checks to tracker bounding boxes."""
        x, y, w, h = bbox
        fh, fw = frame_shape

        area_ratio = (w * h) / (fw * fh)
        aspect = w / h if h > 0 else 0

        # Always apply max-area check
        if area_ratio > TRACKER_MAX_AREA_RATIO:
            return False

        # After confirmation also apply lower bounds and aspect ratio checks
        if self.tracking_state.confirmed:
            if area_ratio < TRACKER_MIN_AREA_RATIO:
                return False
            if aspect > TRACKER_MAX_ASPECT_RATIO or aspect < TRACKER_MIN_ASPECT_RATIO:
                return False

        # Edge hugging check (after a few frames)
        margin_x = fw * TRACKER_EDGE_MARGIN_RATIO
        margin_y = fh * TRACKER_EDGE_MARGIN_RATIO

        if self.tracking_state.frames_tracked > 3:
            if (
                x <= margin_x
                or y <= margin_y
                or x + w >= fw - margin_x
                or y + h >= fh - margin_y
            ):
                return False

        return True
