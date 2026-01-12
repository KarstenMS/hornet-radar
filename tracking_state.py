from typing import Optional, Tuple
import time

class TrackingState:
    def __init__(self):
        # --- Core state ---
        self.active: bool = False
        self.tracker = None

        # --- Geometry ---
        self.bbox: Optional[Tuple[int, int, int, int]] = None

        # --- Counters ---
        self.frames_tracked: int = 0

        # --- Detection control ---
        self.detection_done: bool = False

        # --- Timing ---
        self.start_frame_ts = None
        self.end_frame_ts = None

    # =====================
    # Lifecycle
    # =====================
    def start(self, tracker, bbox):
        self.tracker = tracker
        self.bbox = bbox
        self.active = True
        self.frames_tracked = 0
        self.detection_done = False
        self.centers = []
        self.confirmed = False      # YOLO hat Objekt bestätigt
        self.first_detection_frame = None
        self.last_update_time = None
        self.start_frame_ts = time.time()
        self.end_frame_ts = time.time()
        self.last_update = time.time()


    def update(self, bbox):
        self.bbox = bbox
        self.frames_tracked += 1
        x, y, w, h = bbox
        cx = x + w / 2
        cy = y + h / 2
        self.centers.append((cx, cy))
        self.end_frame_ts = time.time()
        self.dwell_time = self.end_frame_ts - self.start_frame_ts
        self.last_update = time.time()

    def reset(self):
        self.active = False
        self.tracker = None
        self.bbox = None
        self.frames_tracked = 0
        self.detection_done = False
        self.start_frame_ts = None
        self.end_frame_ts = None
        self.centers = []

    # =====================
    # Helper
    # =====================
    def is_stable(self, min_frames: int) -> bool:
        return self.frames_tracked >= min_frames

    def is_timed_out(self, timeout_s: float) -> bool:
        return self.active and (time.time() - self.last_update) > timeout_s
