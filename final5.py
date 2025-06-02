import onnxruntime as ort
import numpy as np
import cv2
from astra_camera import Camera
from can_sender import can_send_pi
from collections import deque

# Camera và model
cam = Camera()
session = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])

WIDTH, HEIGHT = 320, 320
DESIRED_DIST = 1.0  # m

# Trung bình trượt
class MovingAverage:
    def __init__(self, size=5):
        self.q = deque(maxlen=size)
    def update(self, value):
        self.q.append(value)
        return sum(self.q) / len(self.q) if self.q else value

ma_dist = MovingAverage(5)
ma_cx = MovingAverage(5)

# Tiền xử lý
def preprocess(image):
    img = cv2.resize(image, (320, 320))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1) / 255.0
    return np.expand_dims(img.astype(np.float32), axis=0)

# Hậu xử lý
def postprocess(output, conf_thres=0.4, class_id=0):
    preds = output[0]
    boxes = []
    for det in preds:
        conf = det[4] * np.max(det[5:])
        cls = np.argmax(det[5:])
        if conf > conf_thres and cls == class_id:
            x, y, w, h = det[:4]
            x1, y1 = x - w / 2, y - h / 2
            x2, y2 = x + w / 2, y + h / 2
            boxes.append([int(x1), int(y1), int(x2), int(y2)])
    return boxes

frame_count = 0
target = None

try:
    while True:
        frame_count += 1
        color = cam.get_color()
        depth = cam.get_depth()

        # Chỉ detect mỗi 3 frame
        if frame_count % 3 == 0:
            input_tensor = preprocess(color)
            ort_inputs = {session.get_inputs()[0].name: input_tensor}
            output = session.run(None, ort_inputs)
            boxes = postprocess(output)

            min_dist = float("inf")
            target = None
            for box in boxes:
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if 0 <= cx < WIDTH and 0 <= cy < HEIGHT:
                    raw_dist = depth[cy, cx] / 1000.0
                    if 0.1 < raw_dist < min_dist:
                        min_dist = raw_dist
                        target = (cx, cy, raw_dist)

        if target:
            cx, cy, raw_dist = target
            smoothed_dist = ma_dist.update(raw_dist)
            smoothed_cx = ma_cx.update(cx)

            angle = -(smoothed_cx - WIDTH // 2) / (WIDTH // 2) * 90 + 90
            angle = max(0, min(180, angle))

            roi = depth[180:240, 140:180]
            mean_depth = roi.mean() / 1000.0

            move_cmd = 7 if smoothed_dist > 0.6 and mean_depth > 0.4 else 0
            can_send_pi(move_cmd, int(angle), smoothed_dist)
        else:
            can_send_pi(0, 90, 0.0)

except KeyboardInterrupt:
    print("Dừng bởi người dùng.")
