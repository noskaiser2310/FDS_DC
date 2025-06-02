import cv2
from ultralytics import YOLO
from astra_camera import Camera
import numpy as np
from can_sender import can_send_pi 
model = YOLO("yolov8n.pt")  # YOLO detect person
FRAME_WIDTH, FRAME_HEIGHT = 320, 320
FRAME_CENTER_X = FRAME_WIDTH // 2

cam = Camera()

distance_m = 0
print("Starting main loop...")
while True:
    try:
        color,depth = cam.get_depth_and_color()
    except Exception as e:
        print(f"Error getting frames: {e}")
        # time.sleep(1)
        continue
    color_resized = cv2.cvtColor(color_resized, cv2.COLOR_RGB2BGR)
    color_resized = cv2.resize(color, (FRAME_WIDTH, FRAME_HEIGHT))
    results = model.predict(color_resized, verbose=False)
    boxes = results[0].boxes

    for box in boxes:
        if int(box.cls[0]) == 0:  
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            area = min((x2 - x1) * (y2 - y1), (FRAME_HEIGHT * FRAME_WIDTH))
            if area > largest_area and area > 0:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                if 0 <= cy < FRAME_HEIGHT and 0 <= cx < FRAME_WIDTH:
                    distance_m = depth[cy, cx]/1000.0        
                    target_box_info = (cx, cy, area, distance_m)
                    largest_area = area
            if target_box_info:
                cx,cy,area , distance_m = target_box_info
                angle = 90
                try:
                    angle = -(x1-160)/160*90+90
                except:
                    dpt = 0
                can_send_pi(7 *(distance_m > 0.6),-int(angle),distance_m)
    
    