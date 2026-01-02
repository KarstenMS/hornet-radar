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
    return predictions

def count_species(predictions):
    """Count Asian and European hornets based on class IDs."""
    ah_count = sum(1 for p in predictions if p[-1] == 1)
    eh_count = sum(1 for p in predictions if p[-1] == 0)
    return ah_count, eh_count

    