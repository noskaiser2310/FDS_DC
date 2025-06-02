import time
import serial
import onnxruntime as ort
import numpy as np
import cv2
from astra_camera import Camera

# UART đến ESP32
ser = serial.Serial('/dev/serial0', 115200, timeout=1)

# Camera + model
cam = Camera()
session = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])

WIDTH, HEIGHT = 640, 480  # Kích thước khung hình
DESIRED_DIST = 1.0

# Kalman filter
class Kalman1D:
    def __init__(self, q=0.01, r=0.1):
        self.q, self.r, self.x, self.p = q, r, 0, 1
    def update(self, z):
        self.p += self.q
        k = self.p / (self.p + self.r)
        self.x += k * (z - self.x)
        self.p *= (1 - k)
        return self.x

# PID controller
class PID:
    def __init__(self, Kp, Kd):
        self.Kp, self.Kd, self.prev = Kp, Kd, 0
    def compute(self, err):
        d = err - self.prev
        self.prev = err
        return self.Kp * err + self.Kd * d

kf = Kalman1D()
pid_turn = PID(Kp=0.04, Kd=0.01)
pid_move = PID(Kp=1.0, Kd=0.2)

def send(cmd): ser.write((cmd + '\n').encode())

# Tiền xử lý ảnh đầu vào cho ONNX
def preprocess(image):
    img = cv2.resize(image, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1) / 255.0
    return np.expand_dims(img.astype(np.float32), axis=0)

# Hậu xử lý đầu ra để lấy bounding boxes
def postprocess(output, conf_thres=0.4, class_id=0):
    preds = output[0]  # shape: (num, 85)
    boxes = []
    for det in preds:
        conf = det[4] * np.max(det[5:])
        cls = np.argmax(det[5:])
        if conf > conf_thres and cls == class_id:
            x, y, w, h = det[:4]
            x1, y1 = x - w/2, y - h/2
            x2, y2 = x + w/2, y + h/2
            boxes.append([int(x1), int(y1), int(x2), int(y2)])
    return boxes

try:
    while True:
        color = cam.get_color()
        depth = cam.get_depth()

        input_tensor = preprocess(color)
        ort_inputs = {session.get_inputs()[0].name: input_tensor}
        output = session.run(None, ort_inputs)
        boxes = postprocess(output)

        target = None
        min_dist = float("inf")
        for box in boxes:
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if 0 <= cx < WIDTH and 0 <= cy < HEIGHT:
                raw = depth[cy, cx] / 1000.0
                if 0 < raw < min_dist:
                    min_dist = raw
                    target = (cx, cy, raw)

        if target:
            cx, cy, raw = target
            dist = kf.update(raw)
            x_err = cx - WIDTH // 2
            d_err = dist - DESIRED_DIST
            turn = pid_turn.compute(x_err)
            move = pid_move.compute(d_err)

            turn_cmd = "LEFT" if turn < -15 else "RIGHT" if turn > 15 else "STRAIGHT"
            move_cmd = "BACKWARD" if move < -0.3 else "FORWARD" if move > 0.3 else "STOP"

            roi = depth[180:240, 140:180]
            if roi.mean() / 1000.0 < 0.4:
                move_cmd = "STOP"

            send(f"{move_cmd}_{turn_cmd}")
        else:
            send("STOP")

        time.sleep(0.03)

except KeyboardInterrupt:
    ser.close()
