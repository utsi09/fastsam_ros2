#!/usr/bin/env python3
"""
FastSAM ROS2 Node
Subscribes to camera images and performs real-time segmentation
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from collections import deque

from ultralytics import FastSAM


class FastSAMNode(Node):
    """ROS2 Node for FastSAM real-time segmentation"""

    def __init__(self):
        super().__init__('fastsam_node')

        # Parameters
        self.declare_parameter('input_topic', '/camera/left/image')
        self.declare_parameter('output_topic', '/fastsam/result')
        self.declare_parameter('model_path', 'FastSAM-x.pt') #FastSAM-s.pt , 
        self.declare_parameter('imgsz', 640) # 디폴트는 640
        self.declare_parameter('conf', 0.4)
        self.declare_parameter('iou', 0.9)
        self.declare_parameter('device', 'cuda')

        # Get parameters
        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.model_path = self.get_parameter('model_path').value
        self.imgsz = self.get_parameter('imgsz').value
        self.conf = self.get_parameter('conf').value
        self.iou = self.get_parameter('iou').value
        self.device = self.get_parameter('device').value

        # CV Bridge
        self.bridge = CvBridge()

        # Load FastSAM model
        self.get_logger().info(f'Loading FastSAM model: {self.model_path}')
        self.model = FastSAM(self.model_path)
        self.get_logger().info('FastSAM model loaded successfully')

        # Hz measurement
        self.inference_times = deque(maxlen=50)  # Store last 50 inference times
        self.callback_count = 0
        self.last_log_time = time.time()

        # Subscriber
        self.subscription = self.create_subscription(
            Image,
            self.input_topic,
            self.image_callback,
            10
        )

        # Publisher
        self.publisher = self.create_publisher(
            Image,
            self.output_topic,
            10
        )

        self.get_logger().info(f'FastSAM Node initialized')
        self.get_logger().info(f'  Input topic: {self.input_topic}')
        self.get_logger().info(f'  Output topic: {self.output_topic}')
        self.get_logger().info(f'  Image size: {self.imgsz}')
        self.get_logger().info(f'  Device: {self.device}')

    def image_callback(self, msg):
        """Callback when new image is received"""

        # Start timing
        start_time = time.time()

        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Run FastSAM inference
            results = self.model(
                cv_image,
                device=self.device,
                retina_masks=True,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                verbose=False
            )

            # Get inference time
            inference_time = (time.time() - start_time) * 1000  # ms
            self.inference_times.append(inference_time)

            # Get result image with annotations
            result_img = results[0].plot()

            # Publish result
            result_msg = self.bridge.cv2_to_imgmsg(result_img, encoding='bgr8')
            result_msg.header = msg.header
            self.publisher.publish(result_msg)

            # Update callback count
            self.callback_count += 1

            # Log Hz every second
            current_time = time.time()
            if current_time - self.last_log_time >= 1.0:
                self.log_statistics()
                self.last_log_time = current_time

        except Exception as e:
            self.get_logger().error(f'Error in image_callback: {e}')

    def log_statistics(self):
        """Log Hz and inference time statistics"""

        if len(self.inference_times) == 0:
            return

        # Calculate statistics
        avg_time = np.mean(self.inference_times)
        min_time = np.min(self.inference_times)
        max_time = np.max(self.inference_times)
        std_time = np.std(self.inference_times)

        avg_hz = 1000 / avg_time if avg_time > 0 else 0
        max_hz = 1000 / min_time if min_time > 0 else 0
        min_hz = 1000 / max_time if max_time > 0 else 0

        # Log
        self.get_logger().info(
            f'[Hz] Avg: {avg_hz:.2f} Hz | '
            f'Max: {max_hz:.2f} Hz | '
            f'Min: {min_hz:.2f} Hz | '
            f'Time: {avg_time:.2f}±{std_time:.2f} ms | '
            f'Count: {self.callback_count}'
        )


def main(args=None):
    """Main function"""

    rclpy.init(args=args)

    node = FastSAMNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
