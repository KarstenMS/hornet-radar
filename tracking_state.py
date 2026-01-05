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
        self.start_time: float = 0.0
        self.last_update: float = 0.0

    # =====================
    # Lifecycle
    # =====================
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

    # =====================
    # Helper
    # =====================
    def is_stable(self, min_frames: int) -> bool:
        return self.active and self.frames_tracked >= min_frames

    def is_timed_out(self, timeout_s: float) -> bool:
        return self.active and (time.time() - self.last_update) > timeout_s
