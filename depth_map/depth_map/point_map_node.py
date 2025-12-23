#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2

class PointCloudRelayNode(Node):
    def __init__(self):
        super().__init__('point_map_node')

        # rviz에서 사용할 frame을 파라미터로 선언
        self.declare_parameter('frame_id', 'depth_camera_link')
        self.frame_id = self.get_parameter('frame_id').value

        # 구독할 토픽
        self.sub = self.create_subscription(
            PointCloud2,
            '/k4a/points2',
            self.callback,
            10)

        # 재퍼블리시할 토픽
        self.pub = self.create_publisher(
            PointCloud2,
            'points2_visual',
            10)

        self.get_logger().info(f'Subscribing to /k4a/points2, will re-publish on /points2_visual with frame_id="{self.frame_id}"')

    def callback(self, msg: PointCloud2):
        # header.frame_id를 고정된 값으로 덮어쓰기
        msg.header.frame_id = self.frame_id
        # (필요하다면 변환이나 필터링 로직을 넣을 수 있습니다)
        self.pub.publish(msg)
        self.get_logger().debug(f'Relayed {msg.height * msg.width} points')

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudRelayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
