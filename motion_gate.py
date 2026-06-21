# motion_gate.py
import cv2
import time
import logging
from config import (
    FRAME_SKIP,
    MATCH_IOU_THRESHOLD,
    MATCH_MAX_DISTANCE_RATIO,
    MAX_COAST_FRAMES,
    MAX_YOLO_ATTEMPTS,
    MODEL_NAME,
    MOTION_DOWNSCALE,
    MOTION_HISTORY,
    MOTION_KERNEL_SIZE,
    MOTION_MIN_AREA,
    MOTION_VAR_THRESHOLD,
    PI_ID,
    TRACKER_INIT_MAX_AREA_RATIO,
    TRACKER_MIN_AREA_RATIO,
    TRACKING_STABLE_FRAMES,
    YOLO_MATCH_IOU,
    YOLO_RETRY_INTERVAL_FRAMES,
)
from detection import load_model, run_detection
from track import Track
from matching import match, iou, center_distance
from event import DetectionEvent
from sources import FrameSource
from motion_vectors import vector_from_points
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)


class MotionGate:
    """Main stateful pipeline.

    Responsibilities:
        - motion detection (MOG2), run on every camera frame
        - multi-object tracking by associating motion boxes to persistent tracks
        - trigger YOLO per track once it is stable, to confirm species
        - create a <DetectionEvent> for each confirmed track when it ends

    The public API is <process_frame>. It returns a *list* of events because
    several tracks can finalize on the same frame.
    """

    def __init__(self) -> None:
        # --- YOLO ---
        self.model = load_model()

        # --- Tracking ---
        self.tracks: List[Track] = []
        self._next_id = 0
        self.frame_count = 0
        self.source_name = ""

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

    def process_frame(self, frame, source: FrameSource) -> Tuple[List[DetectionEvent], Dict]:
        """Process one frame and return (events, debug)."""
        debug: Dict = {
            "source": source.value,
            "motion": False,
            "tracking": False,
            "yolo_ran": False,
            "motion_boxes": [],
            "tracks": [],
            "fps": None,
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

    def _process_camera(self, frame, debug: Dict) -> Tuple[List[DetectionEvent], Dict]:
        """Camera mode: per-frame motion detection + multi-track + per-track YOLO."""
        now = time.time()
        dt = now - self.last_time
        if dt > 0:
            self.fps = 1.0 / dt
        self.last_time = now
        debug["fps"] = self.fps
        self.frame_count += 1

        # Motion detection runs on EVERY frame now (no appearance tracker to
        # bridge the gaps). It is cheap because MOG2 runs on a downscaled frame.
        motion_boxes = self._update_motion(frame, debug)

        # Associate boxes to tracks, spawn/coast/kill, collect finalized events.
        events = self._update_tracks(frame, motion_boxes, debug)

        # Confirm stable, still-unconfirmed tracks (one shared YOLO call/frame).
        self._maybe_run_yolo(frame, debug)

        return events, debug

    def _process_video(self, frame, debug: Dict) -> Tuple[List[DetectionEvent], Dict]:
        """Video mode: run YOLO on every Nth frame (no tracking)."""
        self.frame_count += 1
        if self.frame_count % FRAME_SKIP != 0:
            return [], debug

        detections = run_detection(frame, self.model)
        if not detections:
            return [], debug

        debug["yolo_ran"] = True
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
        return [event], debug

    def _process_image(self, frame, debug: Dict) -> Tuple[List[DetectionEvent], Dict]:
        """Image mode: run YOLO once (no tracking)."""
        detections = run_detection(frame, self.model)
        if not detections:
            return [], debug

        debug["yolo_ran"] = True
        event = DetectionEvent(
            pi_id=PI_ID,
            detections=detections,
            tracking_bbox=None,
            tracking_frames=0,
            model_name=MODEL_NAME,
            source=self.source_name,
            frame_shape=frame.shape[:2],
        )
        return [event], debug

    def _update_motion(self, frame, debug: Dict):
        """Update background model and return motion bounding boxes (full-res coords)."""
        scale = MOTION_DOWNSCALE
        if scale != 1.0:
            small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            small = frame

        fg = self.bg_subtractor.apply(small)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.kernel)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        inv = 1.0 / scale
        boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # Scale the box back up to full-resolution coordinates.
            x, y, w, h = int(x * inv), int(y * inv), int(w * inv), int(h * inv)
            if w * h < MOTION_MIN_AREA:
                continue
            boxes.append((x, y, w, h))

        debug["motion"] = bool(boxes)
        debug["motion_boxes"] = boxes
        return boxes

    def _update_tracks(self, frame, motion_boxes, debug: Dict) -> List[DetectionEvent]:
        """Greedy-associate motion boxes to tracks; spawn/coast/kill; finalize events."""
        fh, fw = frame.shape[:2]
        max_distance = fw * MATCH_MAX_DISTANCE_RATIO

        track_boxes = [t.bbox for t in self.tracks]
        matches, unmatched_tracks, unmatched_dets = match(
            track_boxes,
            motion_boxes,
            iou_threshold=MATCH_IOU_THRESHOLD,
            max_distance=max_distance,
        )

        for ti, di in matches:
            self.tracks[ti].matched(motion_boxes[di])

        for ti in unmatched_tracks:
            t = self.tracks[ti]
            t.missed()
            self._log_miss(t, motion_boxes, max_distance)

        # Unmatched boxes -> new tracks (with size sanity gating).
        spawned = 0
        for di in unmatched_dets:
            box = motion_boxes[di]
            if not self._spawn_allowed(box, (fh, fw)):
                logger.debug("Box %s rejected for spawn (size gate)", box)
                continue
            self.tracks.append(Track(id=self._next_id, bbox=box, frame_shape=(fh, fw)))
            logger.debug("Track %d SPAWNED at %s", self._next_id, box)
            self._next_id += 1
            spawned += 1

        # Kill coasted-out tracks; finalize the confirmed ones into events.
        events: List[DetectionEvent] = []
        survivors: List[Track] = []
        killed = 0
        for t in self.tracks:
            if t.misses > MAX_COAST_FRAMES:
                killed += 1
                if t.confirmed:
                    events.append(self._finalize_track(t))
                    logger.debug(
                        "Track %d ENDED confirmed=%s after %d frames -> event",
                        t.id, t.confirmed_label, t.frames_tracked,
                    )
                else:
                    logger.debug(
                        "Track %d DROPPED (unconfirmed) after %d frames, coasted out (%d misses)",
                        t.id, t.frames_tracked, t.misses,
                    )
            else:
                survivors.append(t)
        self.tracks = survivors

        # Compact per-frame summary (only when something is going on).
        if motion_boxes or self.tracks or killed:
            logger.debug(
                "frame %d | motion=%d matched=%d spawned=%d killed=%d | active=[%s]",
                self.frame_count, len(motion_boxes), len(matches), spawned, killed,
                ", ".join(self._track_tag(t) for t in self.tracks),
            )

        debug["tracking"] = bool(self.tracks)
        debug["tracks"] = [
            {
                "id": t.id,
                "bbox": tuple(map(int, t.bbox)),
                "confirmed": t.confirmed,
                "label": t.confirmed_label,
                "conf": t.confirmed_confidence,
                "coasting": t.misses > 0,
            }
            for t in self.tracks
        ]
        return events

    def _maybe_run_yolo(self, frame, debug: Dict) -> None:
        """Run a single YOLO inference and confirm any stable tracks it covers."""
        # Tracks that exhausted their attempts without a hit: stop retrying.
        for t in self.tracks:
            if not t.detection_done and not t.confirmed and t.yolo_attempts >= MAX_YOLO_ATTEMPTS:
                t.detection_done = True
                logger.debug("Track %d gave up YOLO confirmation", t.id)

        candidates = [
            t
            for t in self.tracks
            if t.needs_yolo(TRACKING_STABLE_FRAMES, YOLO_RETRY_INTERVAL_FRAMES, MAX_YOLO_ATTEMPTS)
        ]
        if not candidates:
            return

        detections = run_detection(frame, self.model)
        debug["yolo_ran"] = True
        for t in candidates:
            t.yolo_attempts += 1
            t.last_yolo_at_frames_tracked = t.frames_tracked

        logger.debug("YOLO ran for %d candidate track(s): %d detection(s)", len(candidates), len(detections))
        if not detections:
            return

        # Assign each candidate track the detection that best overlaps its box,
        # so a track is confirmed by the insect it is actually following -- not
        # by whichever insect in the frame happens to score highest.
        for t in candidates:
            best_det = None
            best_iou = 0.0
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                det_box = (x1, y1, x2 - x1, y2 - y1)
                ov = iou(t.bbox, det_box)
                if ov > best_iou:
                    best_iou = ov
                    best_det = det

            if best_det is None or best_iou < YOLO_MATCH_IOU:
                continue

            label = "AH" if best_det.get("class_id") == 1 else "EH"
            t.confirmed = True
            t.detection_done = True
            t.confirmed_label = label
            t.confirmed_confidence = float(best_det.get("confidence", 0.0))
            t.confirmed_frame = frame
            t.confirmed_frame_shape = frame.shape
            t.confirmed_yolo_bbox = best_det["bbox"]
            t.detections = [best_det]
            logger.debug("Track %d confirmed as %s (%.2f, IoU=%.2f)", t.id, label, t.confirmed_confidence, best_iou)

    def _finalize_track(self, t: Track) -> DetectionEvent:
        """Create a <DetectionEvent> from a finished, confirmed track."""
        approach_vec = vector_from_points(t.centers, mode="approach")
        departure_vec = vector_from_points(t.centers, mode="departure")

        return DetectionEvent(
            pi_id=PI_ID,
            detections=t.detections,
            model_name=MODEL_NAME,
            source=self.source_name,
            frame=t.confirmed_frame,
            tracking_bbox=t.confirmed_yolo_bbox,
            tracking_frames=len(t.centers),
            frame_shape=(t.confirmed_frame_shape[:2] if t.confirmed_frame_shape is not None else None),
            approach_vec=approach_vec,
            departure_vec=departure_vec,
            dwell_time=t.dwell_time,
        )

    def _log_miss(self, track: Track, motion_boxes, max_distance) -> None:
        """Explain why a track found no match this frame (gate diagnostics)."""
        if not logger.isEnabledFor(logging.DEBUG):
            return
        if not motion_boxes:
            logger.debug("Track %d miss #%d: no motion boxes this frame", track.id, track.misses)
            return
        dists = [center_distance(track.bbox, b) for b in motion_boxes]
        best = min(range(len(motion_boxes)), key=lambda k: dists[k])
        logger.debug(
            "Track %d miss #%d: nearest box dist=%.0f (gate=%.0f), best IoU=%.2f -> %s",
            track.id, track.misses, dists[best], max_distance,
            max(iou(track.bbox, b) for b in motion_boxes),
            "out of range" if dists[best] > max_distance else "below IoU gate",
        )

    def _track_tag(self, t: Track) -> str:
        """Short label for a track in the per-frame summary, e.g. '12:AH' or '13?coast'."""
        if t.confirmed:
            state = t.confirmed_label
        else:
            state = "coast" if t.misses > 0 else "new"
        return f"{t.id}:{state}"

    def _spawn_allowed(self, box, frame_shape) -> bool:
        """Size sanity gate before creating a new track from a motion box."""
        x, y, w, h = box
        fh, fw = frame_shape
        area_ratio = (w * h) / (fw * fh)
        if area_ratio > TRACKER_INIT_MAX_AREA_RATIO:
            return False
        if area_ratio < TRACKER_MIN_AREA_RATIO:
            return False
        return True
