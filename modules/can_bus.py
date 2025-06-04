import can
import struct
import socket
from config import *

class CANSender:
    def __init__(self):
        self.bus = None
        self._init_can_bus()
    
    def _init_can_bus(self):
        try:
            self.bus = can.interface.Bus(
                channel=CAN_CHANNEL, 
                bustype='socketcan', 
                bitrate=CAN_BITRATE
            )
            print("✅ CAN Bus connected")
        except Exception as e:
            print(f"❌ CAN Bus failed: {e}")
            self.bus = None
    
    def send_control(self, speed, angle):
        if self.bus is None:
            print(f"⚠️  CAN not available - Speed: {speed}, Angle: {angle}")
            return False
        
        try:
            # Chuyển đổi và gửi
            speed_be = socket.htons(int(speed))
            angle_be = socket.htons(int(angle))
            data = struct.pack('>hh', speed_be, angle_be)
            
            msg = can.Message(arbitration_id=CAN_MSG_ID, data=data, is_extended_id=False)
            self.bus.send(msg)
            return True
            
        except Exception as e:
            print(f"❌ CAN send error: {e}")
            return False 