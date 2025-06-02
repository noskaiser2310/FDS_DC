import can
import struct
import socket  # Dùng hton/ntoh
bus = None
try:
    bus = can.interface.Bus(channel='can0', bustype='socketcan', bitrate=500000) 
    # Hoặc ví dụ cho PCAN: bus = can.interface.Bus(bustype='pcan', channel='PCAN_USBBUS1', bitrate=500000)
    print("CAN bus initialized successfully.")
except Exception as e:
    print(f"Failed to initialize CAN bus: {e}")
    print("CAN messages will not be sent.")

# Giá trị CAN thực tế
CAN_SPEED_NEUTRAL = 1500
CAN_SPEED_FORWARD_INCREMENT = 50

def can_send_pi(speed, angle, distance_to_target): 
    # Chuyển đổi sang kiểu int vì pack và htons yêu cầu
    speed = int(round(speed))
    angle = int(round(angle))

    print(f'CAN Send -> Speed In: {speed}, Steer Adjust In: {angle}, Dist: {distance_to_target:.2f}m')
    print(f'           -> CAN Values: Speed={speed}, Steer={angle}')
    
    if bus: 
        try:
            speed_be = socket.htons(speed)
            steer_be = socket.htons(angle)

            data = struct.pack('>hh', speed_be, steer_be)
            
            msg = can.Message(arbitration_id=0x21, data=data, is_extended_id=False)
            bus.send(msg)

        except Exception as e:
            print(f"Error sending CAN message: {e}")
    else:
        print("CAN bus not available. Message not sent.")