# image_recognition.py
from detection import load_model, run_detection

class ImageRecognition:
    def __init__(self):
        self.model = load_model()

    def detect(self, image):
        """
        Run object detection on a single image.
        Returns raw detections.
        """
        return run_detection(image, self.model)
