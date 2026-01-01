import os
import sys
import time
#import datetime
#import platform
from picamera2 import Picamera2
import cv2
from vibe import BackgroundSubtractor  # nur wenn du Bewegungserkennung nutzt
import torch
from ultralytics import YOLO
from yolov5.models.yolo import DetectionModel

torch.serialization.add_safe_globals([DetectionModel])



# ======== Parameter und Setup ========

ROOT = "/opt/vespai"
MODEL_PATH = os.path.join(ROOT, "yolov5/models/yolov5s-all-data.pt")
YOLO_DIR = os.path.join(ROOT, "yolov5")

CONF = 0.8               # Konfidenzschwelle
FRAME_DELAY = 50
SAVE = True
SAVE_DIR = os.path.join(ROOT, "monitor/detections")
os.makedirs(SAVE_DIR, exist_ok=True)

# ======== Kamera mit Picamera2 ========

picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(main={"size": (1920, 1080)})
)
picam2.start()
time.sleep(1)  # kleine Wartezeit, bis Kamera stabil ist

# ======== YOLOv5 laden ========
sys.path.insert(0, YOLO_DIR)
os.chdir(YOLO_DIR)
model = torch.hub.load(YOLO_DIR, "custom", path=MODEL_PATH, source="local", device="cpu")
model.conf = CONF

# ======== Hauptschleife ========

frame_id = 0
print("Kamera gestartet. Drücke ESC zum Beenden.")

while True:
    # Capture Frame als NumPy-Array
    frame = picam2.capture_array()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #convert colorspace from BGR to RGB

    # YOLOv5-Inferenz
    results = model(frame_rgb)
    predictions = results.pred[0]
    print(predictions)
    # Zähle Hornissen
    ah_count, eh_count = 0, 0
    for p in predictions:
        if p[-1] == 1:
            ah_count += 1
        elif p[-1] == 0:
            eh_count += 1

    # Wenn Hornissen erkannt wurden
    if ah_count + eh_count > 0:
        print(f"🚨 Frame #{frame_id}: {ah_count}x Vespa velutina, {eh_count}x Vespa crabro")
        results.render()
        annotated = cv2.cvtColor(results.ims[0], cv2.COLOR_RGB2BGR)

        # Optional speichern
        if SAVE:
            fname = os.path.join(SAVE_DIR, f"frame-{frame_id}.jpg")
            cv2.imwrite(fname, annotated)

    # Frame anzeigen
    cv2.imshow("Hornet Detector (IMX500)", frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC-Taste
        break

    frame_id += 1
    time.sleep(0.2)  # kleine Pause zwischen Frames

cv2.destroyAllWindows()
picam2.stop()
print("Beendet.")
