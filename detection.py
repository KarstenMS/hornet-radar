"""
Handles YOLO detection and tracking logic.
"""

import torch
from ultralytics import YOLO
import cv2
import warnings
from config import YOLO_DIR, MODEL_DIR, CONFIDENCE_THRESHOLD

warnings.filterwarnings("ignore", category=FutureWarning) # For suppressing Torch FutureWarnings

def load_model():
    """Load YOLOv5 model from local directory."""
    model = torch.hub.load(YOLO_DIR, "custom", path=MODEL_DIR, source="local")
    model.conf = CONFIDENCE_THRESHOLD
    return model

def run_detection(image, model):
    """Run YOLO detection on a single image."""
    results = model(image)
    predictions = results.pred[0]
    results.render()
    return parse_predictions(predictions, results)

def count_species(predictions):
    ah_count = sum(1 for p in predictions if p["class_id"] == 1)
    eh_count = sum(1 for p in predictions if p["class_id"] == 0)
    return ah_count, eh_count


def parse_predictions(predictions):
    parsed = []
    for p in predictions:
        x1, y1, x2, y2, conf, cls = p.tolist()
        parsed.append({
            "bbox": (int(x1), int(y1), int(x2), int(y2)),
            "confidence": conf,
            "class_id": int(cls)
        })
    return parsed
