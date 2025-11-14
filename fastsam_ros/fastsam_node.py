"""
FastSAM ROS2 Node - Show only inference time (ms)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time
from ultralytics import YOLO


class FastSAMNode(Node):
    """ROS2 Node for YOLO segmentation (log inference time only)"""

    def __init__(self):
        super().__init__('fastsam_node')

        # Parameters
        self.declare_parameter('input_topic', '/carla/hero/rgb_front/image')
        self.declare_parameter('output_topic', '/fastsam/result')
        self.declare_parameter('model_path', 'yolo11n-seg.pt') # yolo11n.pt yolo11n-seg.pt yolov8n-seg.pt FastSAM-x.pt FastSAM-s.pt
        self.declare_parameter('imgsz', 1280) # 96x96 >100~150ms   |  192x192 >150~180ms | 320x320 > 200~250ms | 640x640(디폴트) > 500ms(실시간불가능)
        self.declare_parameter('conf', 0.4)
        self.declare_parameter('iou', 0.9)
        self.declare_parameter('device', 'cuda')

        # Get params
        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.model_path = self.get_parameter('model_path').value
        self.imgsz = self.get_parameter('imgsz').value
        self.conf = self.get_parameter('conf').value
        self.iou = self.get_parameter('iou').value
        self.device = self.get_parameter('device').value

        # CV Bridge
        self.bridge = CvBridge()

        # Load model
        self.get_logger().info(f'Loading YOLO segmentation model: {self.model_path}')
        self.model = YOLO(self.model_path)
        self.get_logger().info('YOLO model loaded successfully')

        # ROS I/O
        self.subscription = self.create_subscription(
            Image, self.input_topic, self.image_callback, 10)
        self.publisher = self.create_publisher(Image, self.output_topic, 10)

        self.get_logger().info(f'FastSAM Node initialized on {self.device}')

    def image_callback(self, msg):
        """Callback when image received"""
        start_time = time.time()

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Run YOLO segmentation
            results = self.model(
                cv_image,
                device=self.device,
                retina_masks=True,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                verbose=False
            )

            # 추론 완료 후 시간 계산
            inference_time = (time.time() - start_time) * 1000  # ms
            self.get_logger().info(f"Inference time: {inference_time:.2f} ms")

            # 결과 이미지 출력 (선택)
            result_img = results[0].plot()
            result_msg = self.bridge.cv2_to_imgmsg(result_img, encoding='bgr8')
            result_msg.header = msg.header
            self.publisher.publish(result_msg)

        except Exception as e:
            self.get_logger().error(f'Error in image_callback: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = FastSAMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
