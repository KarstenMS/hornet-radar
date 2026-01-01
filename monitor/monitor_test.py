import torch
from ultralytics import YOLO
import os
from yolov5.models.yolo import Detectionmodel

torch.serialization.add_safe_globals([DetectionModel])

ROOT = "/home/hornet1"



MODEL_PATH = os.path.join(ROOT, "hornet-radar/models/yolov5s-all-data.pt")
YOLO_DIR = os.path.join(ROOT, "yolov5")

model = torch.hub.load(YOLO_DIR, "custom", path=MODEL_PATH, source="local")
#model = torch.hub.load("ultralytics/yolov5", "yolov5s")

results = model("https://ultralytics.com/images/zidane.jpg")

print(results)