"""Hornet Radar: YOLO model loading and inference utilities."""

import logging
import warnings
from typing import Any, Dict, List
import cv2
import torch
from config import YOLO_DIR, MODEL_DIR, YOLO_CONF_THRESHOLD, YOLO_IMG_SIZE, YOLO_TORCH_THREADS

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning) # For suppressing Torch FutureWarnings

def load_model():
    """Load a YOLOv5 model from a local clone via torch.hub.

    Returns:
        A torch.hub-loaded YOLO model instance.

    Notes:
        This expects YOLOv5 source code to be available at YOLO_DIR and a weights file at MODEL_DIR.
    """
    # Cap CPU threads so inference does not starve the real-time capture/tracking
    # loop (see YOLO_TORCH_THREADS). Must be set before/around model use.
    if YOLO_TORCH_THREADS:
        torch.set_num_threads(YOLO_TORCH_THREADS)

    model = torch.hub.load(YOLO_DIR, "custom", path=MODEL_DIR, source="local")
    model.conf = YOLO_CONF_THRESHOLD
    return model

def run_detection(image, model) -> List[Dict[str, Any]]:
    """Run YOLO detection on a single image.

    Args:
        image: A numpy array (OpenCV image).
        model: The loaded YOLO model.

    Returns:
        A list of dicts: {bbox, confidence, class_id}.
    """
    # YOLOv5's hub AutoShape expects RGB for numpy input; frames throughout
    # the pipeline are BGR (cv2 / Picamera2 "RGB888" / cv2.VideoCapture).
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(rgb, size=YOLO_IMG_SIZE)
    predictions = results.pred[0]

    return parse_predictions(predictions)

def parse_predictions(predictions) -> List[Dict[str, Any]]:
    """Convert raw YOLO predictions to a serializable list."""
    parsed: List[Dict[str, Any]] = []
    for p in predictions:
        x1, y1, x2, y2, conf, cls = p.tolist()
        parsed.append({
            "bbox": (int(x1), int(y1), int(x2), int(y2)),
            "confidence": float(conf),
            "class_id": int(cls)
        })
    return parsed



