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

    # --- Flight segmentation (for clean approach/departure vectors) ---
    sitting: bool = False                                        # currently judged to be sitting still at the bait
    approach_centers: Optional[List[Tuple[float, float]]] = None # snapshot of the arrival flight, frozen at the first sit
    departure_start: Optional[int] = None                        # index into `centers` where the exit flight begins (set when motion resumes after a sit)

    # --- Presence check (feeding bout) ---
    departed: bool = False                                       # YOLO no longer finds the insect at the box -> it has left
    presence_failures: int = 0                                   # consecutive failed presence checks

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
    last_yolo_frame: int = -1                                    # global frame index of the last YOLO run for this track (confirmation or presence)

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
        # Motion after a sit means the insect is leaving: mark where the exit
        # flight starts so the departure vector uses only the exit segment and
        # never bleeds into the (unrelated) arrival flight.
        if self.sitting and self.departure_start is None:
            self.departure_start = len(self.centers)
        self.sitting = False

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

    def mark_present(self) -> None:
        """A presence check confirmed the insect is still sitting at the box.

        Keeps the track alive across the feeding bout without appending a center
        or advancing frames_tracked (the insect is not moving).
        """
        self.misses = 0
        now = time.time()
        self.dwell_time = now - self.start_frame_ts
        self.last_update = now

    def is_stationary_at_bait(self, edge_margin_ratio: float) -> bool:
        """Whether this track likely sits at the (centrally-placed) bait.

        The signal that an insect landed is that it STOPPED producing motion
        (MOG2 no longer emits a box, so the track is coasting) while its last
        position is well inside the frame. An insect leaving instead keeps
        producing motion toward an edge, so edge-proximity rules that out.
        Speed is irrelevant once coasting: a sitting insect makes no motion at
        all, and its last tracked step (the landing approach) may still be fast.
        """
        if self.frame_shape is None or self.misses <= 0:
            return False
        fh, fw = self.frame_shape
        cx, cy = _center(self.bbox)
        mx, my = fw * edge_margin_ratio, fh * edge_margin_ratio
        return mx <= cx <= fw - mx and my <= cy <= fh - my

    def freeze_approach(self) -> None:
        """Snapshot the arrival flight, once, at the first sit.

        At this point `centers` holds only the arrival flight (a sit appends no
        centers), so this snapshot is a clean approach that later departure
        motion cannot bleed into.
        """
        if self.approach_centers is None:
            self.approach_centers = list(self.centers)


def _center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0
