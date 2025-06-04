import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import threading
from config import *

class SimpleCamera:
    def __init__(self):
        if not rclpy.ok():
            rclpy.init()
        
        self.node = Node('simple_camera')
        self.bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None
        self.lock = threading.Lock()
        
        # Subscribe to topics
        self.node.create_subscription(Image, CAMERA_RGB_TOPIC, self._rgb_callback, 10)
        self.node.create_subscription(Image, CAMERA_DEPTH_TOPIC, self._depth_callback, 10)
        
        # Start ROS thread
        self.ros_thread = threading.Thread(target=lambda: rclpy.spin(self.node))
        self.ros_thread.daemon = True
        self.ros_thread.start()
    
    def _rgb_callback(self, msg):
        with self.lock:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
    
    def _depth_callback(self, msg):
        with self.lock:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")
    
    def get_images(self):
        with self.lock:
            return self.rgb_image, self.depth_image
    
    def cleanup(self):
        self.node.destroy_node() 