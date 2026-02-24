"""Hornet Radar: YOLO model loading and inference utilities."""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List

import torch

from config import CONFIDENCE_THRESHOLD, MODEL_DIR, YOLO_DIR

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress Torch FutureWarnings


def load_model():
    """Load a YOLOv5 model from a local clone via torch.hub.

    Returns:
        A torch.hub-loaded YOLO model instance.

    Notes:
        This expects YOLOv5 source code to be available at YOLO_DIR and a weights file at MODEL_DIR.
    """
    model = torch.hub.load(YOLO_DIR, "custom", path=MODEL_DIR, source="local")
    model.conf = CONFIDENCE_THRESHOLD
    return model


def run_detection(image, model) -> List[Dict[str, Any]]:
    """Run YOLO detection on a single image.

    Args:
        image: A numpy array (OpenCV image).
        model: The loaded YOLO model.

    Returns:
        A list of dicts: {bbox, confidence, class_id}.
    """
    results = model(image)
    predictions = results.pred[0]
    return parse_predictions(predictions)


def parse_predictions(predictions) -> List[Dict[str, Any]]:
    """Convert raw YOLO predictions to a serializable list."""
    parsed: List[Dict[str, Any]] = []
    for p in predictions:
        x1, y1, x2, y2, conf, cls = p.tolist()
        parsed.append(
            {
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "confidence": float(conf),
                "class_id": int(cls),
            }
        )
    return parsed
