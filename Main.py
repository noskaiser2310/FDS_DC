import time
from camera import Camera
from fn_model import MModel
from can_sender import can_send_pi

DEFAULT_SPEED = 1500
DEFAULT_ANGLE = 90
SLEEP_INTERVAL = 0.1  # Gửi CAN mỗi 0.1s
DESIRED_DISTANCE = 1.0

def main():
    print("[INFO] Starting main loop with AI + CAN control.")
    cam = Camera()
    model = MModel("best.onnx", width=320, height=320)
    frame_count = 0

    # Khởi tạo giá trị mặc định
    speed = DEFAULT_SPEED
    angle = DEFAULT_ANGLE

    try:
        while True:
            rgb = cam.get_rgb_image()
            depth = cam.get_depth_image()

            if rgb is not None and depth is not None:
                frame_count += 1

                if frame_count % 3 == 0:
                    boxes = model.infer(rgb)
                    if boxes:
                        # chọn box lớn nhất
                        largest = max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
                        speed, angle, dist = model.calculate_control(largest, depth)
                        print(f"[AI] speed={speed}, angle={angle}, distance={dist:.2f}m")
                    else:
                        print("[AI] No person detected → dùng điều khiển mặc định")
                        speed = DEFAULT_SPEED
                        angle = DEFAULT_ANGLE

            # Gửi CAN dù có hoặc không có kết quả AI
            can_send_pi(speed, angle, 0.0)
            time.sleep(SLEEP_INTERVAL)

    except KeyboardInterrupt:
        print("\n[EXIT] Stopped by user.")

    finally:
        cam.destroy()
        print("[CLEANUP] Camera shutdown.")

if __name__ == "__main__":
    main()
