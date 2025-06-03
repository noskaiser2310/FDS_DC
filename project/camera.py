import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import threading
import time

class Camera:
    """
    A Python class representing a ROS 2 camera, providing methods
    to retrieve RGB and depth images in OpenCV format.
    """
    def __init__(self, node_name='camera_data_reader'):
        """
        Initializes the Camera class.
        Creates an internal ROS 2 node and sets up subscribers for color and depth images.
        """
        # Check if rclpy has been initialized; if not, initialize it
        if not rclpy.ok():
            rclpy.init()

        self.node = Node(node_name)
        self.node.get_logger().info(f'Camera instance node "{node_name}" has been started.')

        self.bridge = CvBridge()

        self._rgb_image = None
        self._depth_image = None
        self._rgb_lock = threading.Lock()
        self._depth_lock = threading.Lock()

        # Subscriber for color image
        self.rgb_subscription = self.node.create_subscription(
            Image,
            '/camera/color/image_raw',
            self._rgb_callback,
            10  # QoS: queue size
        )
        self.node.get_logger().info('Subscribing to /camera/color/image_raw')

        # Subscriber for depth image
        self.depth_subscription = self.node.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self._depth_callback,
            10  # QoS: queue size
        )
        self.node.get_logger().info('Subscribing to /camera/depth/image_raw')

        # Separate thread to spin the ROS 2 node
        self._ros_thread = threading.Thread(target=self._spin_node)
        self._ros_thread.daemon = True # Ensure the thread exits when the main program exits
        self._ros_thread.start()

    def _spin_node(self):
        """
        Internal function to spin the ROS 2 node in a separate thread.
        """
        rclpy.spin(self.node)
        self.node.get_logger().info('ROS 2 node spinning stopped.')

    def _rgb_callback(self, msg):
        """
        Callback for the color image topic.
        Converts the ROS Image message to OpenCV format (BGR8).
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self._rgb_lock:
                self._rgb_image = cv_image
        except Exception as e:
            self.node.get_logger().error(f'Error converting RGB image: {e}')
            with self._rgb_lock:
                self._rgb_image = None

    def _depth_callback(self, msg):
        """
        Callback for the depth image topic.
        Converts the ROS Image message to OpenCV format (passthrough).
        """
        try:
            # Use "passthrough" to retain the original format of the depth image (e.g., 16UC1, 32FC1)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            with self._depth_lock:
                self._depth_image = cv_image
        except Exception as e:
            self.node.get_logger().error(f'Error converting Depth image: {e}')
            with self._depth_lock:
                self._depth_image = None

    def get_rgb_image(self):
        """
        Returns the latest RGB image in OpenCV format (numpy array).
        Returns None if no image is available or an error occurred.
        """
        with self._rgb_lock:
            return self._rgb_image

    def get_depth_image(self):
        """
        Returns the latest depth image in OpenCV format (numpy array).
        Returns None if no image is available or an error occurred.
        """
        with self._depth_lock:
            return self._depth_image

    def destroy(self):
        """
        Cleans up ROS 2 node resources.
        Should be called when the Camera object is no longer needed.
        """
        self.node.destroy_node()
        # rclpy.shutdown() # Do not call shutdown here if you intend to have multiple ROS nodes or instances
        self.node.get_logger().info('Camera instance node destroyed.')

