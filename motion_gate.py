# motion_gate.py
import cv2
import time
import logging
from config import (
    FRAME_SKIP,
    MATCH_IOU_THRESHOLD,
    MATCH_MAX_DISTANCE_RATIO,
    STATIONARY_EDGE_MARGIN_RATIO,
    MAX_COAST_FRAMES,
    MAX_COAST_FRAMES_STATIONARY,
    MAX_YOLO_ATTEMPTS,
    MODEL_NAME,
    MOTION_DOWNSCALE,
    MOTION_HISTORY,
    MOTION_KERNEL_SIZE,
    MOTION_CLOSE_KERNEL_SIZE,
    MOTION_MIN_AREA_RATIO,
    MOTION_VAR_THRESHOLD,
    PI_ID,
    TRACKER_INIT_MAX_AREA_RATIO,
    TRACKER_MIN_AREA_RATIO,
    TRACKING_STABLE_FRAMES,
    YOLO_MATCH_IOU,
    YOLO_MIN_AREA_RATIO,
    YOLO_RETRY_INTERVAL_FRAMES,
    YOLO_PRESENCE_INTERVAL_FRAMES,
    YOLO_PRESENCE_LOST_LIMIT,
    YOLO_IMG_SIZE,
    YOLO_PRESENCE_IMG_SIZE,
)
from yolo_worker import YoloWorker
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
        # --- YOLO (runs in a background thread; see YoloWorker) ---
        self.yolo = YoloWorker()

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
        # Larger kernel to CLOSE gaps so one insect's parts merge into one box.
        self.close_kernel = (
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (MOTION_CLOSE_KERNEL_SIZE, MOTION_CLOSE_KERNEL_SIZE)
            )
            if MOTION_CLOSE_KERNEL_SIZE
            else None
        )

        # --- FPS (camera only) ---
        self.last_time = time.time()
        self.fps = 0.0

    def close(self) -> None:
        """Stop the background YOLO worker. Safe to call once at shutdown."""
        self.yolo.stop()

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

        # Apply any finished background YOLO inference to the (living) tracks
        # BEFORE the kill pass, so mark_present/departed affect this frame.
        self._apply_yolo_results(debug)

        # Associate boxes to tracks, spawn/coast/kill, collect finalized events.
        events = self._update_tracks(frame, motion_boxes, debug)

        # Queue a new YOLO inference if a track needs confirmation or a presence
        # check (non-blocking; the worker runs it in the background).
        self._submit_yolo(frame, debug)

        return events, debug

    def _process_video(self, frame, debug: Dict) -> Tuple[List[DetectionEvent], Dict]:
        """Video mode: run YOLO on every Nth frame (no tracking)."""
        self.frame_count += 1
        if self.frame_count % FRAME_SKIP != 0:
            return [], debug

        detections = self.yolo.detect_sync(frame)
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
        detections = self.yolo.detect_sync(frame)
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
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.kernel)   # drop speckle
        if self.close_kernel is not None:
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.close_kernel)  # merge one insect's parts

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        inv = 1.0 / scale
        fh, fw = frame.shape[:2]
        min_area = MOTION_MIN_AREA_RATIO * fw * fh  # resolution-independent gate
        boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # Scale the box back up to full-resolution coordinates.
            x, y, w, h = int(x * inv), int(y * inv), int(w * inv), int(h * inv)
            if w * h < min_area:
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

        taken = {di for _, di in matches}
        for ti in unmatched_tracks:
            t = self.tracks[ti]
            t.missed()
            self._log_miss(t, motion_boxes, max_distance, taken)

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
            stationary = t.is_stationary_at_bait(STATIONARY_EDGE_MARGIN_RATIO)
            if stationary:
                # Freeze the arrival flight the first time it lands, and mark it
                # sitting so the next matched box is recognised as the exit flight.
                t.freeze_approach()
                t.sitting = True

            # A track that YOLO has judged departed dies quickly (short budget) so
            # its event finalizes right after it leaves. A track still sitting at
            # the central bait is kept alive long (the presence check refreshes it
            # every frame it is seen); anything else uses the normal short budget.
            if t.departed:
                coast_budget = MAX_COAST_FRAMES
            elif stationary:
                coast_budget = MAX_COAST_FRAMES_STATIONARY
            else:
                coast_budget = MAX_COAST_FRAMES

            if t.misses > coast_budget:
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

        frame_area = float(fw * fh)
        debug["tracking"] = bool(self.tracks)
        debug["tracks"] = [
            {
                "id": t.id,
                "bbox": tuple(map(int, t.bbox)),
                "confirmed": t.confirmed,
                "label": t.confirmed_label,
                "conf": t.confirmed_confidence,
                "coasting": t.misses > 0,
                "sitting": t.sitting,
                "departed": t.departed,
                "area_pct": 100.0 * (t.bbox[2] * t.bbox[3]) / frame_area,
            }
            for t in self.tracks
        ]
        return events

    def _submit_yolo(self, frame, debug: Dict) -> None:
        """Queue one background YOLO inference if a track needs it and the worker
        is free. Non-blocking: results are applied later in _apply_yolo_results.

        Two purposes:
        - confirm still-unconfirmed tracks that are stable or have just landed;
        - presence-check confirmed tracks sitting at the bait, to keep them alive
          across the feeding bout and detect when they depart.
        """
        fc = self.frame_count

        # Unconfirmed tracks that exhausted their attempts without a hit: give up.
        for t in self.tracks:
            if not t.detection_done and not t.confirmed and t.yolo_attempts >= MAX_YOLO_ATTEMPTS:
                t.detection_done = True
                logger.debug("Track %d gave up YOLO confirmation", t.id)

        if self.yolo.busy():
            return  # an inference is already in flight; retry once it returns

        confirm = [t for t in self.tracks if self._needs_confirmation(t, fc)]
        presence = [t for t in self.tracks if self._needs_presence_check(t, fc)]
        if not confirm and not presence:
            return

        # Snapshot the candidate boxes so the result is matched against the frame
        # it was computed on, not the (by then moved) current boxes. Use the full
        # resolution when a species must be confirmed; otherwise this is a cheap
        # presence check for a sitting insect, which runs at a small size.
        track_boxes = [(t.id, tuple(t.bbox)) for t in confirm + presence]
        size = YOLO_IMG_SIZE if confirm else YOLO_PRESENCE_IMG_SIZE
        if not self.yolo.submit(frame.copy(), track_boxes, size):
            return  # worker became busy between the check and the submit

        for t in confirm:
            t.yolo_attempts += 1
        for t in confirm + presence:
            t.last_yolo_frame = fc
        logger.debug(
            "YOLO submitted: %d confirm, %d presence candidate(s)",
            len(confirm), len(presence),
        )

    def _apply_yolo_results(self, debug: Dict) -> None:
        """Apply finished background YOLO results to the still-living tracks.

        Detections are matched against each track's snapshot box (from submit
        time) and applied to the live track by id; tracks that died while the
        inference ran are simply skipped.
        """
        results = self.yolo.poll()
        if not results:
            return
        by_id = {t.id: t for t in self.tracks}
        for track_boxes, frame, detections in results:
            debug["yolo_ran"] = True
            logger.debug(
                "YOLO result: %d requested box(es), %d detection(s)",
                len(track_boxes), len(detections),
            )
            for tid, snap_bbox in track_boxes:
                t = by_id.get(tid)
                if t is None:
                    continue  # track died while the inference was running

                best_det, best_iou = self._best_overlap(snap_bbox, detections)
                hit = best_det is not None and best_iou >= YOLO_MATCH_IOU

                if not t.confirmed:
                    # Confirmation: attach the best-overlapping detection, so the
                    # track is confirmed by the insect it is actually following.
                    if hit:
                        self._apply_detection(t, best_det, frame)
                        logger.debug(
                            "Track %d confirmed as %s (%.2f, IoU=%.2f)",
                            t.id, t.confirmed_label, t.confirmed_confidence, best_iou,
                        )
                else:
                    # Presence: still at its box? keep alive + refresh proof image;
                    # gone for YOLO_PRESENCE_LOST_LIMIT checks in a row -> departed.
                    if hit:
                        t.mark_present()
                        t.presence_failures = 0
                        self._apply_detection(t, best_det, frame)
                        logger.debug("Track %d still present (IoU=%.2f)", t.id, best_iou)
                    else:
                        t.presence_failures += 1
                        logger.debug(
                            "Track %d presence check failed (%d/%d)",
                            t.id, t.presence_failures, YOLO_PRESENCE_LOST_LIMIT,
                        )
                        if t.presence_failures >= YOLO_PRESENCE_LOST_LIMIT:
                            t.departed = True
                            logger.debug("Track %d marked DEPARTED (presence lost)", t.id)

    def _needs_confirmation(self, t: Track, fc: int) -> bool:
        """Whether an unconfirmed track should be offered to YOLO this frame."""
        if t.detection_done or t.confirmed:
            return False
        if t.yolo_attempts >= MAX_YOLO_ATTEMPTS:
            return False
        # Don't spend a YOLO inference on a track too small to be a hornet
        # (wasps/flies): they otherwise burn inferences and get mis-confirmed.
        if t.frame_shape is not None:
            fh, fw = t.frame_shape
            _, _, w, h = t.bbox
            if w * h < YOLO_MIN_AREA_RATIO * fw * fh:
                return False
        # Ready once it has tracked enough frames OR has just landed at the bait
        # (a sharp, motionless insect is the ideal moment to identify it).
        ready = t.frames_tracked >= TRACKING_STABLE_FRAMES or t.is_stationary_at_bait(
            STATIONARY_EDGE_MARGIN_RATIO
        )
        if not ready:
            return False
        if t.yolo_attempts > 0 and (fc - t.last_yolo_frame) < YOLO_RETRY_INTERVAL_FRAMES:
            return False
        return True

    def _needs_presence_check(self, t: Track, fc: int) -> bool:
        """Whether a confirmed, still-sitting track should be presence-checked."""
        if not t.confirmed or t.departed:
            return False
        if not t.is_stationary_at_bait(STATIONARY_EDGE_MARGIN_RATIO):
            return False
        return (fc - t.last_yolo_frame) >= YOLO_PRESENCE_INTERVAL_FRAMES

    @staticmethod
    def _best_overlap(bbox, detections) -> Tuple:
        """Return (detection, iou) of the detection best overlapping `bbox`."""
        best_det = None
        best_iou = 0.0
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            det_box = (x1, y1, x2 - x1, y2 - y1)
            ov = iou(bbox, det_box)
            if ov > best_iou:
                best_iou = ov
                best_det = det
        return best_det, best_iou

    @staticmethod
    def _apply_detection(t: Track, det: Dict, frame) -> None:
        """Confirm a track from a detection (also used to refresh the proof image)."""
        t.confirmed = True
        t.detection_done = True
        t.confirmed_label = "AH" if det.get("class_id") == 1 else "EH"
        t.confirmed_confidence = float(det.get("confidence", 0.0))
        t.confirmed_frame = frame
        t.confirmed_frame_shape = frame.shape
        t.confirmed_yolo_bbox = det["bbox"]
        t.detections = [det]

    def _finalize_track(self, t: Track) -> DetectionEvent:
        """Create a <DetectionEvent> from a finished, confirmed track.

        Approach uses the arrival flight (snapshotted when the insect first sat,
        so a feeding sit does not let the exit flight bleed into it); departure
        uses only the exit segment after that sit. For a pure fly-by (never sat)
        both fall back to the single continuous trajectory.
        """
        approach_points = t.approach_centers if t.approach_centers is not None else t.centers
        departure_points = t.centers[t.departure_start:] if t.departure_start is not None else t.centers
        approach_vec = vector_from_points(approach_points, mode="approach")
        departure_vec = vector_from_points(departure_points, mode="departure")

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

    def _log_miss(self, track: Track, motion_boxes, max_distance, taken) -> None:
        """Explain why a track found no match this frame (gate diagnostics).

        Distinguishes the three real causes: no motion at all, every box out of
        the matching gate, or a matchable box that another track claimed first
        (greedy contention).
        """
        if not logger.isEnabledFor(logging.DEBUG):
            return
        if not motion_boxes:
            logger.debug("Track %d miss #%d: no motion boxes this frame", track.id, track.misses)
            return

        stats = [
            (center_distance(track.bbox, b), iou(track.bbox, b), i)
            for i, b in enumerate(motion_boxes)
        ]
        dist, ov, _ = min(stats, key=lambda s: s[0])  # nearest box
        in_gate = [s for s in stats if s[0] <= max_distance or s[1] >= MATCH_IOU_THRESHOLD]

        if not in_gate:
            reason = "out of range"
        elif all(s[2] in taken for s in in_gate):
            reason = "matchable box taken by another track"
        else:
            reason = "lost greedy contention"

        logger.debug(
            "Track %d miss #%d: nearest dist=%.0f (gate=%.0f) IoU=%.2f -> %s",
            track.id, track.misses, dist, max_distance, ov, reason,
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
