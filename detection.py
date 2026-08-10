"""Hornet Radar: YOLO model loading and inference utilities."""

import logging
import os
import warnings
from typing import Any, Dict, List
import cv2
import numpy as np
import torch
from config import (
    YOLO_DIR,
    MODEL_DIR,
    YOLO_CONF_THRESHOLD,
    YOLO_IMG_SIZE,
    YOLO_TORCH_THREADS,
    YOLO_ZOOM_OUT,
    YOLO_DEBUG_DIR,
)

logger = logging.getLogger(__name__)
_debug_dump_count = 0  # sequential index for YOLO_DEBUG_DIR dumps
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

def run_detection(image, model, size: int = None, zoom: float = None) -> List[Dict[str, Any]]:
    """Run YOLO detection on a single image.

    Args:
        image: A numpy array (OpenCV image).
        model: The loaded YOLO model.
        size: Inference resolution (longest side). Defaults to YOLO_IMG_SIZE;
            a smaller value (e.g. for cheap presence checks) is much faster.
        zoom: Zoom-out factor (see YOLO_ZOOM_OUT). Defaults to the config value;
            pass 1.0 to disable (e.g. for the scale test). The frame is shrunk by
            1/zoom and padded to full size so insects appear smaller (closer to
            the model's training scale); detections are mapped back to the
            original image coordinates.

    Returns:
        A list of dicts: {bbox, confidence, class_id}.
    """
    # YOLOv5's hub AutoShape expects RGB for numpy input; frames throughout
    # the pipeline are BGR (cv2 / Picamera2 "RGB888" / cv2.VideoCapture).
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    zoom = YOLO_ZOOM_OUT if zoom is None else zoom
    remap = None
    if zoom and zoom > 1.0:
        rgb, remap = _zoom_out(rgb, zoom)

    results = model(rgb, size=size or YOLO_IMG_SIZE)
    predictions = results.pred[0]
    detections = parse_predictions(predictions)

    # Save exactly what YOLO received (the zoomed frame) with its raw boxes, to
    # verify the apparent size / scores match what scale_test.py predicts.
    if YOLO_DEBUG_DIR:
        _dump_yolo_input(rgb, detections, size or YOLO_IMG_SIZE, zoom)

    if remap is not None:
        detections = _remap_detections(detections, remap, image.shape[:2])
    return detections


def _dump_yolo_input(rgb, detections, size, zoom) -> None:
    """Write the (zoomed) YOLO input with its detections drawn, for diagnosis."""
    global _debug_dump_count
    try:
        os.makedirs(YOLO_DEBUG_DIR, exist_ok=True)
        vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        fh, fw = vis.shape[:2]
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]  # canvas coords = what YOLO saw
            label = "AH" if d.get("class_id") == 1 else "EH"
            conf = float(d.get("confidence", 0.0))
            area_pct = 100.0 * (x2 - x1) * (y2 - y1) / (fw * fh)
            color = (0, 0, 255) if label == "AH" else (0, 200, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, f"{label} {conf:.2f} {area_pct:.1f}%", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(vis, f"size={size} zoom={zoom}", (10, fh - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        _debug_dump_count += 1
        cv2.imwrite(os.path.join(YOLO_DEBUG_DIR, f"yolo_{_debug_dump_count:05d}.jpg"), vis)
    except Exception:
        logger.exception("YOLO debug dump failed")


def _zoom_out(rgb, zoom: float):
    """Shrink `rgb` by 1/zoom and pad back to its size with the background colour.

    Returns (canvas, (pad_x, pad_y, zoom)); an object then occupies 1/zoom^2 of
    the frame area, as if the camera were `zoom` times farther away.
    """
    h, w = rgb.shape[:2]
    sw, sh = max(1, round(w / zoom)), max(1, round(h / zoom))
    small = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA)

    # Fill = median of a subsampled frame (dominated by the green backdrop), so
    # the padding looks like more background rather than black bars.
    fill = np.median(rgb[::8, ::8].reshape(-1, 3), axis=0)
    canvas = np.full((h, w, 3), fill, dtype=rgb.dtype)

    pad_x, pad_y = (w - sw) // 2, (h - sh) // 2
    canvas[pad_y:pad_y + sh, pad_x:pad_x + sw] = small
    return canvas, (pad_x, pad_y, zoom)


def _remap_detections(detections, remap, orig_hw) -> List[Dict[str, Any]]:
    """Map bboxes from the zoomed-out canvas back to original image coordinates."""
    pad_x, pad_y, zoom = remap
    h, w = orig_hw
    out = []
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        x1 = (x1 - pad_x) * zoom
        y1 = (y1 - pad_y) * zoom
        x2 = (x2 - pad_x) * zoom
        y2 = (y2 - pad_y) * zoom
        # Clip into the original frame; drop boxes that fall in the padding.
        x1 = max(0, min(w, x1)); x2 = max(0, min(w, x2))
        y1 = max(0, min(h, y1)); y2 = max(0, min(h, y2))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        nd = dict(d)
        nd["bbox"] = (int(x1), int(y1), int(x2), int(y2))
        out.append(nd)
    return out

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



