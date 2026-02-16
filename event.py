# event.py
import uuid
import time
from helpers import timestamp
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
        source: str,
        tracking_bbox: Optional[Tuple[int, int, int, int]] = None,
        tracking_frames: int = 0,
        frame_shape: Optional[Tuple[int, int]] = None,
        approach_vec: Optional[Tuple[int, int]] = None,
        departure_vec: Optional[Tuple[int, int]] = None,
        dwell_time: Optional[float] = None,
        frame=None,
    ):
        # === Identity ===
        self.event_id: str = str(uuid.uuid4())
        self.pi_id: str = pi_id
        self.timestamp: timestamp = timestamp()
        self.source: str = source

        # === Detection results ===
        self.detections: List[Dict] = detections
        self.model_name: str = model_name
        self.frame = frame

        # === Confidence ===
        self.confidence = max([d["confidence"] for d in detections], default=0.0)

        # === Tracking context ===
        self.tracking_bbox = tracking_bbox
        self.tracking_frames = tracking_frames
        self.frame_shape = frame_shape

        # === Approach & Departure vectors ===
        self.trajectory: List[Tuple[int, int]] = []
        self.approach_vec = approach_vec
        self.departure_vec = departure_vec
        self.dwell_time = dwell_time
        self.metadata: Dict = {}

        # === Media URLs & Dir ===
        self.image_path = None
        self.thumb_path= None
        self.image_url = None
        self.thumb_url= None

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
            "tracking_frames": self.tracking_frames,
            "frame_shape": self.frame_shape,
            "trajectory": self.trajectory,
            "approach_vec": self.approach_vec,
            "departure_vec": self.departure_vec,
            "dwell_time": self.dwell_time,
            "metadata": self.metadata,
            "source": self.source,
            "image_url": self.image_url,
            "thumb_url": self.thumb_url,

        }

    def __repr__(self) -> str:
        return (
            f"<DetectionEvent {self.event_id[:8]} "
            f"pi={self.pi_id} "
            f"detections={len(self.detections)} "
            f"confidence={self.confidence:.2f}>"
        )
