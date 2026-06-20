"""Hornet Radar: per-object track for the motion-box multi-tracker.

A <Track> is the multi-object replacement for the old single-object TrackingState.
Each detected insect gets one Track with a stable id that persists across frames.
Identity is maintained purely by associating motion boxes to existing tracks
(see <matching>); no OpenCV appearance tracker is involved, which keeps the
pipeline cheap enough to run motion detection on every frame on a Pi 5.
"""
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple


@dataclass
class Track:
    """State of a single tracked object (one insect)."""

    # --- Identity ---
    id: int

    # --- Geometry ---
    bbox: Tuple[float, float, float, float]                      # last known (x, y, w, h)
    centers: List[Tuple[float, float]] = field(default_factory=list)
    frame_shape: Optional[Tuple[int, int]] = None

    # --- Counters ---
    frames_tracked: int = 0
    misses: int = 0                                              # consecutive frames without a matched box (coasting)
    frames_since_confirmed: int = 0

    # --- Timing ---
    start_frame_ts: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    dwell_time: float = 0.0

    # --- YOLO / Confirmation ---
    confirmed: bool = False
    confirmed_label: Optional[str] = None
    confirmed_confidence: Optional[float] = None
    detection_done: bool = False                                # confirmed OR gave up after MAX_YOLO_ATTEMPTS
    yolo_attempts: int = 0
    last_yolo_at_frames_tracked: int = 0

    confirmed_frame: Any = None
    confirmed_frame_shape: Any = None
    confirmed_yolo_bbox: Any = None
    detections: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.centers == []:
            cx, cy = _center(self.bbox)
            self.centers.append((cx, cy))

    def matched(self, bbox: Tuple[float, float, float, float]) -> None:
        """Update the track with a newly associated motion box."""
        self.bbox = bbox
        self.misses = 0
        self.frames_tracked += 1

        cx, cy = _center(bbox)
        self.centers.append((float(cx), float(cy)))

        now = time.time()
        self.dwell_time = now - self.start_frame_ts
        self.last_update = now

        if self.confirmed:
            self.frames_since_confirmed += 1

    def missed(self) -> None:
        """Register a frame in which no motion box matched this track (coasting)."""
        self.misses += 1

    def needs_yolo(self, stable_frames: int, retry_interval: int, max_attempts: int) -> bool:
        """Whether this track should be offered to YOLO for confirmation this frame."""
        if self.detection_done:
            return False
        if self.frames_tracked < stable_frames:
            return False
        if self.yolo_attempts >= max_attempts:
            return False
        if self.yolo_attempts > 0:
            if (self.frames_tracked - self.last_yolo_at_frames_tracked) < retry_interval:
                return False
        return True


def _center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0
