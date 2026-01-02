import torch
from ultralytics import YOLO
import os
import cv2


ROOT = "/home/hornet1/hornet-radar"



MODEL_PATH = os.path.join(ROOT, "model/yolov5s-all-data.pt")
YOLO_DIR = os.path.join(ROOT, "yolov5")

model = torch.hub.load(YOLO_DIR, "custom", path=MODEL_PATH, source="local")


results = model("https://ultralytics.com/images/zidane.jpg")

print(results)

