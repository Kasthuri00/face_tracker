import cv2
import numpy as np
from ultralytics import YOLO

class FaceDetector:
    def __init__(self, model_path="detectors/yolov8n-face.pt"):
        self.model = YOLO(model_path)  # Load the model dynamically

    def detect(self, frame):
        results = self.model(frame)[0]
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            return boxes
        return []
