#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs_py import point_cloud2
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid

class PointcloudTo2DMapNode(Node):
    def __init__(self):
        super().__init__('pointcloud_to_2d_map_node')

        # --- 파라미터 선언 및 읽기 ---
        self.declare_parameter('input_cloud_topic', '/camera/depth/points')
        self.declare_parameter('map_topic', '/pointcloud_2d_map')
        self.declare_parameter('frame_id', 'camera_base')
        self.declare_parameter('resolution', 0.05)  # meters per cell
        self.declare_parameter('width', 200)        # number of cells in X
        self.declare_parameter('height', 200)       # number of cells in Y (ground plane)
        self.declare_parameter('z_min', 0.05)       # ignore points closer than this (meters)
        self.declare_parameter('z_max', 5.0)        # ignore points beyond this

        self.input_topic = self.get_parameter('input_cloud_topic').value
        self.map_topic   = self.get_parameter('map_topic').value
        self.frame_id    = self.get_parameter('frame_id').value
        self.resolution  = self.get_parameter('resolution').value
        self.width       = self.get_parameter('width').value
        self.height      = self.get_parameter('height').value
        self.z_min       = self.get_parameter('z_min').value
        self.z_max       = self.get_parameter('z_max').value

        # --- origin 설정 (그리드의 중앙을 카메라 위치로) ---
        self.origin_x = - (self.width  * self.resolution) / 2.0
        self.origin_y = - (self.height * self.resolution) / 2.0

        self.get_logger().info(
            f"Subscribed to: {self.input_topic}, publishing map: {self.map_topic}\n"
            f"Grid {self.width}×{self.height} @ {self.resolution}m, "
            f"Z-filter [{self.z_min}, {self.z_max}], frame={self.frame_id}"
        )

        # 퍼블리셔/구독자
        self.pub = self.create_publisher(OccupancyGrid, self.map_topic, 10)
        self.sub = self.create_subscription(
            PointCloud2,
            self.input_topic,
            self.cb_pointcloud,
            10
        )

    def cb_pointcloud(self, msg: PointCloud2):
        try:
            gen = point_cloud2.read_points(msg,
                                          field_names=('x','y','z'),
                                          skip_nans=True)
            pts_list = [(float(x), float(y), float(z)) for x, y, z in gen]
            if not pts_list:
                return

            pts = np.array(pts_list, dtype=np.float64)
            if pts.ndim == 1:
                pts = pts.reshape(1,3)

            # Z 축(카메라 정면 축)을 높이로 보고 필터링
            mask = (pts[:,2] >= self.z_min) & (pts[:,2] <= self.z_max)
            pts = pts[mask]

            grid = np.zeros((self.height, self.width), dtype=np.int8)

            if pts.shape[0] > 0:
                # X–Z 평면 투영: X→grid x, Z→grid y
                xs = -pts[:,0]  # camera right +
                ys = -pts[:,2]  # camera forward +

                gx = np.floor((xs - self.origin_x) / self.resolution).astype(int)
                gy = np.floor((ys - self.origin_y) / self.resolution).astype(int)

                valid = (gx >= 0) & (gx < self.width) & (gy >= 0) & (gy < self.height)
                grid[self.height - 1 - gy[valid], gx[valid]] = 100

            occ = OccupancyGrid()
            occ.header.stamp = self.get_clock().now().to_msg()
            occ.header.frame_id = self.frame_id

            info = occ.info
            info.resolution = self.resolution
            info.width      = self.width
            info.height     = self.height
            info.origin.position.x = self.origin_x
            info.origin.position.y = self.origin_y
            info.origin.position.z = 0.0
            info.origin.orientation.w = 1.0  # no rotation

            data = grid.T.flatten().tolist()
            occ.data = [int(v) for v in data]

            self.pub.publish(occ)

        except Exception as e:
            self.get_logger().error(f"Error in cb_pointcloud: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = PointcloudTo2DMapNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
