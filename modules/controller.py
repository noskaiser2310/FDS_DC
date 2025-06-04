import numpy as np
from config import *

class VehicleController:
    def __init__(self):
        self.current_speed = DEFAULT_SPEED
        self.current_angle = DEFAULT_ANGLE
        print("🚗 Vehicle Controller initialized")
    
    def calculate_control(self, detection_box, depth_image):
        if detection_box is None:
            # Không phát hiện người -> giữ nguyên hoặc dừng
            return DEFAULT_SPEED, DEFAULT_ANGLE, 0.0
        
        # Tính vị trí và khoảng cách
        x1, y1, x2, y2, conf = detection_box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Lấy khoảng cách từ depth image
        distance = self._get_distance(center_x, center_y, depth_image)
        
        # Tính toán điều khiển
        speed = self._calculate_speed(distance)
        angle = self._calculate_steering(center_x, depth_image.shape[1])
        
        return speed, angle, distance
    
    def _get_distance(self, x, y, depth_image):
        if depth_image is None:
            return MAX_DISTANCE
        
        h, w = depth_image.shape
        if 0 <= x < w and 0 <= y < h:
            depth_value = depth_image[y, x]
            if isinstance(depth_value, np.uint16):
                return depth_value / 1000.0  # mm -> m
            return float(depth_value)
        return MAX_DISTANCE
    
    def _calculate_speed(self, distance):
        if distance < MIN_DISTANCE:
            return MIN_SPEED  # Dừng khẩn cấp
        elif distance < SAFE_DISTANCE:
            return DEFAULT_SPEED - 200  # Chậm lại
        elif distance > SAFE_DISTANCE * 2:
            return DEFAULT_SPEED + 100  # Tăng tốc nhẹ
        else:
            return DEFAULT_SPEED
    
    def _calculate_steering(self, person_x, frame_width):
        frame_center = frame_width // 2
        offset = person_x - frame_center
        offset_ratio = offset / frame_center
        offset_ratio = max(-1.0, min(1.0, offset_ratio))  # Giới hạn [-1, 1]
        
        angle_adjustment = int(offset_ratio * MAX_ANGLE_OFFSET)
        return DEFAULT_ANGLE + angle_adjustment 