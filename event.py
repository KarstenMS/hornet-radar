# event.py
import time
import uuid
from typing import List, Dict, Optional, Tuple


class DetectionEvent:
    """
    Represents one confirmed detection event.
    An event is created AFTER stable tracking and a YOLO detection run.
    """

    def __init__(
        self,
        pi_id: str,
        detections: List[Dict],
        *,
        model_name: str,
        tracking_bbox: Optional[Tuple[int, int, int, int]] = None,
        roi_bbox: Optional[Tuple[int, int, int, int]] = None,
        tracking_frames: int = 0,
        frame_shape: Optional[Tuple[int, int]] = None,
    ):
        # === Identity ===
        self.event_id: str = str(uuid.uuid4())
        self.pi_id: str = pi_id
        self.timestamp: float = time.time()

        # === Detection results ===
        self.detections: List[Dict] = detections
        self.model_name: str = model_name

        # === Confidence ===
        self.confidence: float = self._compute_confidence(detections)

        # === Tracking context ===
        self.tracking_bbox = tracking_bbox
        self.roi_bbox = roi_bbox
        self.tracking_frames = tracking_frames
        self.frame_shape = frame_shape

        # === Future extensions ===
        self.trajectory: List[Tuple[int, int]] = []
        self.flight_angle: Optional[float] = None
        self.metadata: Dict = {}

    # --------------------------------------------------
    # Confidence handling
    # --------------------------------------------------

    def _compute_confidence(self, detections: List[Dict]) -> float:
        """
        Computes an overall confidence score for the event.

        Strategy (v1):
        - If multiple detections exist:
          → take MAX confidence
        """
        if not detections:
            return 0.0

        confidences = [
            d.get("confidence", 0.0)
            for d in detections
            if isinstance(d.get("confidence"), (int, float))
        ]

        return max(confidences) if confidences else 0.0

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def has_species(self, keyword: str) -> bool:
        """
        Checks if any detection contains the keyword
        (e.g. 'asian', 'vespa_velutina').
        """
        keyword = keyword.lower()
        for d in self.detections:
            label = str(d.get("label", "")).lower()
            if keyword in label:
                return True
        return False

    def to_dict(self) -> Dict:
        """
        Serializes the event for storage / upload.
        """
        return {
            "event_id": self.event_id,
            "pi_id": self.pi_id,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "model": self.model_name,
            "detections": self.detections,
            "tracking_bbox": self.tracking_bbox,
            "roi_bbox": self.roi_bbox,
            "tracking_frames": self.tracking_frames,
            "frame_shape": self.frame_shape,
            "trajectory": self.trajectory,
            "flight_angle": self.flight_angle,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return (
            f"<DetectionEvent {self.event_id[:8]} "
            f"pi={self.pi_id} "
            f"detections={len(self.detections)} "
            f"confidence={self.confidence:.2f}>"
        )
