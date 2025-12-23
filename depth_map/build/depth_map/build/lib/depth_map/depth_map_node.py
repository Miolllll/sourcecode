#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import OccupancyGrid

class DepthMapNode(Node):
    def __init__(self):
        super().__init__('depth_map_node')

        # 파라미터
        self.declare_parameter('frame_id', 'depth_camera_link')
        self.declare_parameter('resolution', 0.05)   # 1셀 크기 [m]
        self.declare_parameter('width', 200)         # 그리드 폭 [셀]
        self.declare_parameter('height', 200)        # 그리드 높이 [셀]
        self.declare_parameter('z_min', 0.05)        # 유효 최소 깊이 [m]
        self.declare_parameter('z_max', 5.0)         # 유효 최대 깊이 [m]

        self.frame_id   = self.get_parameter('frame_id').value
        self.resolution = self.get_parameter('resolution').value
        self.width      = self.get_parameter('width').value
        self.height     = self.get_parameter('height').value
        self.z_min      = self.get_parameter('z_min').value
        self.z_max      = self.get_parameter('z_max').value

        # grid 원점 (카메라 중심 기준)
        self.origin_x = -(self.width  * self.resolution) / 2.0
        self.origin_y = -(self.height * self.resolution) / 2.0

        # CvBridge, intrinsics 준비
        self.bridge = CvBridge()
        self.fx = self.fy = self.cx = self.cy = None

        # 구독: 카메라 정보 & 깊이 영상
        self.create_subscription(CameraInfo,
                                 '/k4a/depth/camera_info',
                                 self.camera_info_cb, 10)
        self.create_subscription(Image,
                                 '/k4a/depth/image_raw',
                                 self.depth_cb, 10)

        # 퍼블리셔: 2D OccupancyGrid
        self.pub = self.create_publisher(OccupancyGrid,
                                         '/depth_occupancy_grid', 10)

        self.get_logger().info('DepthMapNode initialized.')

    def camera_info_cb(self, msg: CameraInfo):
        if self.fx is None:
            self.fx = msg.k[0]; self.fy = msg.k[4]
            self.cx = msg.k[2]; self.cy = msg.k[5]
            self.get_logger().info(f'Got camera intrinsics: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}')

    def depth_cb(self, msg: Image):
        if self.fx is None:
            return   # intrinsics 아직 수신 전

        # 1) 깊이 → 미터 단위 float32
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')

        # 2) 유효 범위 마스킹
        mask = (depth >= self.z_min) & (depth <= self.z_max)
        ys, xs = np.nonzero(mask)
        zs      = depth[ys, xs]

        # 3) 각 화소를 world XY 평면으로 projection
        x = (xs - self.cx) * zs / self.fx      # 카메라 오른쪽(+X)
        y = zs                                 # 카메라 앞쪽(+Y)
        gx = ((x - self.origin_x) / self.resolution).astype(np.int32)
        gy = ((y - self.origin_y) / self.resolution).astype(np.int32)

        valid = (0 <= gx) & (gx < self.width) & (0 <= gy) & (gy < self.height)

        # 4) 매 프레임마다 완전 초기화
        grid = np.zeros((self.height, self.width), dtype=np.int8)
        grid[gy[valid], gx[valid]] = 100

        # 5) OccupancyGrid 메시지 작성·퍼블리시
        occ = OccupancyGrid()
        occ.header.stamp = self.get_clock().now().to_msg()
        occ.header.frame_id = self.frame_id
        occ.info.resolution = self.resolution
        occ.info.width      = self.width
        occ.info.height     = self.height
        occ.info.origin.position.x = self.origin_x
        occ.info.origin.position.y = self.origin_y
        occ.info.origin.orientation.w = 1.0
        occ.data = grid.flatten().tolist()
        self.pub.publish(occ)

def main(args=None):
    rclpy.init(args=args)
    node = DepthMapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
