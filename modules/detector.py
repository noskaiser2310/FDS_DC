import onnxruntime as ort
import cv2
import numpy as np
from config import *

class PersonDetector:
    def __init__(self):
        self.session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        print(f"✅ AI Model loaded: {MODEL_PATH}")
    
    def detect(self, image):
        # Resize và chuẩn bị ảnh
        img = cv2.resize(image, MODEL_INPUT_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1) / 255.0
        input_tensor = np.expand_dims(img.astype(np.float32), axis=0)
        
        # Chạy AI
        ort_inputs = {self.session.get_inputs()[0].name: input_tensor}
        output = self.session.run(None, ort_inputs)
        
        # Xử lý kết quả
        detections = output[0][0].T
        boxes = []
        for det in detections:
            x, y, w, h, conf = det
            if conf > CONFIDENCE_THRESHOLD:
                x1, y1 = int(x - w/2), int(y - h/2)
                x2, y2 = int(x + w/2), int(y + h/2)
                boxes.append([x1, y1, x2, y2, conf])
        
        return boxes
    
    def get_largest_detection(self, boxes):
        if not boxes:
            return None
        return max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1])) 