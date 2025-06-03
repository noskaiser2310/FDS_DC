import onnxruntime as ort
import numpy as np
import cv2

# Các giá trị mặc định có thể cấu hình
DEFAULT_SPEED = 1500
DEFAULT_ANGLE = 90
DESIRED_DISTANCE = 1.0  # m

class MModel:
    def __init__(self, model_path, width=320, height=320):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.width = width
        self.height = height

    def preprocess(self, image):
        img = cv2.resize(image, (self.width, self.height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1) / 255.0
        return np.expand_dims(img.astype(np.float32), axis=0)

    def postprocess(self, output, conf_thres=0.4):
        output = output[0][0].T  # (num_boxes, 5)
        boxes = []
        for det in output:
            x, y, w, h, conf = det
            if float(conf) > conf_thres:
                x1, y1 = int(x - w / 2), int(y - h / 2)
                x2, y2 = int(x + w / 2), int(y + h / 2)
                boxes.append([x1, y1, x2, y2])
        return boxes

    def infer(self, image):
        input_tensor = self.preprocess(image)
        ort_inputs = {self.session.get_inputs()[0].name: input_tensor}
        output = self.session.run(None, ort_inputs)
        return self.postprocess(output)

    def get_box_center(self, box):
        x1, y1, x2, y2 = box
        return (x1 + x2) // 2, (y1 + y2) // 2

    def get_depth_at_center(self, depth_img, box):
        cx, cy = self.get_box_center(box)
        if 0 <= cx < depth_img.shape[1] and 0 <= cy < depth_img.shape[0]:
            d = depth_img[cy, cx]
            if isinstance(d, np.uint16):
                return d / 1000.0  # mm → m
            return float(d)
        return -1.0

    def calculate_control(self, box, depth_image):
        cx, _ = self.get_box_center(box)
        frame_center = depth_image.shape[1] // 2
        offset_ratio = (cx - frame_center) / frame_center
        offset_ratio = max(-1.0, min(1.0, offset_ratio))

        distance = self.get_depth_at_center(depth_image, box)

        speed = DEFAULT_SPEED
        if 0 < distance < 5:
            delta = distance - DESIRED_DISTANCE
            speed += int(delta * 200)

        angle = DEFAULT_ANGLE + int(offset_ratio * 300)
        return speed, angle, distance
