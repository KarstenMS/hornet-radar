"""Hornet Radar: in-memory state container for an ongoing tracking session."""
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

@dataclass
class TrackingState:
    """State of the currently tracked object.

    This class is intentionally a simple container to keep <MotionGate> readable.
    It stores tracking geometry, timestamps, and the result of a (single) YOLO confirmation.
    """

    # --- Core ---
    active: bool = False
    tracker: Any = None

    # --- Geometry ---
    bbox: Optional[Tuple[float, float, float, float]] = None  # (x, y, w, h)
    centers: List[Tuple[float, float]] = field(default_factory=list)
    frame_shape: Optional[Tuple[int, int]] = None

    # --- Counters ---
    frames_tracked: int = 0
    invalid_frames: int = 0
    frames_since_confirmed: int = 0

    # --- Timing ---
    start_frame_ts: Optional[float] = None
    end_frame_ts: Optional[float] = None
    last_update: Optional[float] = None
    dwell_time: float = 0.0

    # --- YOLO / Confirmation ---
    confirmed: bool = False
    confirmed_label: Optional[str] = None
    confirmed_confidence: Optional[float] = None
    detection_done: bool = False
    yolo_attempts: int = 0

    confirmed_frame: Any = None
    confirmed_frame_shape: Any = None
    confirmed_yolo_bbox: Any = None

    detections: list = field(default_factory=list)
    last_good_frame: Any = None
    last_good_frame_shape: Any = None

    def reset(self) -> None:
        """Reset to the initial (inactive) state."""
        # Reinitialize by overwriting fields. (dataclass doesn't provide a built-in reset)
        self.__dict__.update(TrackingState().__dict__)

    def start(self, tracker: Any, bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int]) -> None:
        """Start tracking with the given tracker and initial bounding box."""
        self.tracker = tracker
        self.bbox = bbox
        self.active = True
        self.frames_tracked = 0
        self.invalid_frames = 0
        self.frames_since_confirmed = 0
        self.detection_done = False
        self.centers.clear()

        now = time.time()
        self.start_frame_ts = now
        self.end_frame_ts = now
        self.last_update = now
        self.frame_shape = frame_shape

    def update(self, bbox: Tuple[float, float, float, float]) -> None:
        """Update state with the newest tracker bounding box."""
        self.bbox = bbox
        self.frames_tracked += 1

        x, y, w, h = bbox
        cx = x + w / 2
        cy = y + h / 2
        self.centers.append((float(cx), float(cy)))

        now = time.time()
        self.end_frame_ts = now
        if self.start_frame_ts is not None:
            self.dwell_time = now - self.start_frame_ts
        self.last_update = now
