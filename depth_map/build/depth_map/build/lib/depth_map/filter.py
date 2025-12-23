#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

class PointCloudFilterNode(Node):
    def __init__(self):
        super().__init__('filter_node')

        # --- 파라미터 선언 ---
        self.declare_parameter('input_cloud_topic', '/camera/depth/points')
        self.declare_parameter('filtered_cloud_topic', '/filtered_points')
        self.declare_parameter('x_min', -4.0)
        self.declare_parameter('x_max',  4.0)
        self.declare_parameter('y_min', -1.8)
        self.declare_parameter('y_max', 0.0)
        self.declare_parameter('z_min',  0.5)
        self.declare_parameter('z_max',  5.0)

        # 파라미터 값 읽기
        in_topic  = self.get_parameter('input_cloud_topic').value
        out_topic = self.get_parameter('filtered_cloud_topic').value
        self.x_min = self.get_parameter('x_min').value
        self.x_max = self.get_parameter('x_max').value
        self.y_min = self.get_parameter('y_min').value
        self.y_max = self.get_parameter('y_max').value
        self.z_min = self.get_parameter('z_min').value
        self.z_max = self.get_parameter('z_max').value

        # 토픽 선언 로그
        self.get_logger().info(
            f'Filtering PointCloud: in={in_topic}, out={out_topic}, '
            f'x[{self.x_min},{self.x_max}] '
            f'y[{self.y_min},{self.y_max}] '
            f'z[{self.z_min},{self.z_max}]'
        )

        # 퍼블리셔·구독자 생성
        self.pub = self.create_publisher(PointCloud2, out_topic, 10)
        self.sub = self.create_subscription(PointCloud2, in_topic, self.cb_cloud, 10)

    def cb_cloud(self, msg: PointCloud2):
        pts = list(pc2.read_points(msg, field_names=('x','y','z'), skip_nans=True))
        flt = [(x,y,z) for x,y,z in pts
               if self.x_min <= x <= self.x_max
               and self.y_min <= y <= self.y_max
               and self.z_min <= z <= self.z_max]
        if not flt:
            return
        out_msg = pc2.create_cloud_xyz32(msg.header, flt)
        self.pub.publish(out_msg)
        self.get_logger().debug(f'Published {len(flt)}/{len(pts)} points')

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudFilterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()
