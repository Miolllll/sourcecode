#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math, threading, time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.callback_groups import ReentrantCallbackGroup

from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2


class LatestOnlyFilter(Node):
    def __init__(self):
        super().__init__("point_cloud_filter_node")

        # -------- 파라미터 선언 --------
        self.declare_parameter("input_cloud_topic", "/k4a/points2")
        self.declare_parameter("output_cloud_topic_reliable", "/filtered_points")
        self.declare_parameter("output_cloud_topic_fast", "/filtered_points_fast")
        # 퍼블리셔 on/off 스위치
        self.declare_parameter("publish_reliable", True)   # 기본: RELIABLE만 사용
        self.declare_parameter("publish_fast", False)      # fast는 기본 OFF

        self.declare_parameter("enable_filter", True)
        self.declare_parameter("proc_rate_hz", 30.0)    # 30 Hz 권장 (33ms 예산)
        self.declare_parameter("max_age_ms", 300)       # 오래된 프레임 폐기 임계

        # ROI / FOV
        self.declare_parameter("z_min", 0.2)
        self.declare_parameter("z_max", 3.5)
        self.declare_parameter("y_min", -1.4)
        self.declare_parameter("y_max", -0.15)
        self.declare_parameter("fov_deg", 120.0)        # 전방 시야각 (좌우 합)

        # 보xel
        self.declare_parameter("voxel_size", 0.05)      # 10cm 보xel
        self.declare_parameter("voxel_method", "centroid")  # ["centroid", "first"]
        self.declare_parameter("voxel_offset_m", 20.0)      # 음수좌표 대비 인덱스 오프셋(ROI보다 크게)

        # QoS/타이밍
        self.declare_parameter("sub_best_effort", True)     # 최신 프레임 우선
        self.declare_parameter("warn_budget_ms", 33.0)      # 30Hz 예산

        # -------- 파라미터 읽기 --------
        in_topic  = self.get_parameter("input_cloud_topic").value
        out_rel   = self.get_parameter("output_cloud_topic_reliable").value
        out_fast  = self.get_parameter("output_cloud_topic_fast").value
        self.pub_rel_enable = bool(self.get_parameter("publish_reliable").value)
        self.pub_be_enable  = bool(self.get_parameter("publish_fast").value)

        self.enable_filter = bool(self.get_parameter("enable_filter").value)
        self.proc_rate     = float(self.get_parameter("proc_rate_hz").value)
        self.max_age_ms    = int(self.get_parameter("max_age_ms").value)

        self.z_min   = float(self.get_parameter("z_min").value)
        self.z_max   = float(self.get_parameter("z_max").value)
        self.y_min   = float(self.get_parameter("y_min").value)
        self.y_max   = float(self.get_parameter("y_max").value)
        self.fov_deg = float(self.get_parameter("fov_deg").value)
        self.voxel   = float(self.get_parameter("voxel_size").value)

        self.voxel_method   = str(self.get_parameter("voxel_method").value).lower()
        self.voxel_offset_m = float(self.get_parameter("voxel_offset_m").value)

        self.sub_best_effort = bool(self.get_parameter("sub_best_effort").value)
        self.warn_budget_ms  = float(self.get_parameter("warn_budget_ms").value)

        # 미리 계산
        self.fov_half_tan = math.tan(math.radians(self.fov_deg * 0.5))

        # -------- QoS/콜백 그룹 --------
        if self.sub_best_effort:
            sub_qos = QoSProfile(depth=1)
            sub_qos.reliability = ReliabilityPolicy.BEST_EFFORT
            sub_qos.history = HistoryPolicy.KEEP_LAST
        else:
            sub_qos = QoSProfile(depth=5)
            sub_qos.reliability = ReliabilityPolicy.RELIABLE
            sub_qos.history = HistoryPolicy.KEEP_LAST

        qos_rel = QoSProfile(depth=5)
        qos_rel.reliability = ReliabilityPolicy.RELIABLE
        qos_rel.history = HistoryPolicy.KEEP_LAST

        qos_be = QoSProfile(depth=1)
        qos_be.reliability = ReliabilityPolicy.BEST_EFFORT
        qos_be.history = HistoryPolicy.KEEP_LAST

        self.cb_group = ReentrantCallbackGroup()

        self.sub  = self.create_subscription(
            PointCloud2, in_topic, self._cb_store_latest, sub_qos, callback_group=self.cb_group
        )
        # 퍼블리셔는 스위치에 따라 생성
        self.pub_rel = self.create_publisher(PointCloud2, out_rel, qos_rel) if self.pub_rel_enable else None
        self.pub_be  = self.create_publisher(PointCloud2, out_fast, qos_be) if self.pub_be_enable else None

        self._lock = threading.Lock()
        self._latest_msg = None

        self.timer = self.create_timer(1.0 / self.proc_rate, self._process_latest, callback_group=self.cb_group)

        self.get_logger().info(
            f"[LatestOnlyFilter] in={in_topic} -> "
            f"{'out_rel='+out_rel if self.pub_rel else 'out_rel=DISABLED'}, "
            f"{'out_fast='+out_fast if self.pub_be else 'out_fast=DISABLED'} | "
            f"proc={self.proc_rate}Hz age<={self.max_age_ms}ms | FOV={self.fov_deg} voxel={self.voxel} "
            f"z[{self.z_min},{self.z_max}] y[{self.y_min},{self.y_max}] "
            f"enable_filter={self.enable_filter} sub_best_effort={self.sub_best_effort}"
        )

    # 최신 메시지만 저장 (백로그 방지)
    def _cb_store_latest(self, msg: PointCloud2):
        with self._lock:
            self._latest_msg = msg

    # NumPy 기반 보xel 다운샘플 (셀 평균 또는 첫점)
    def _voxel_downsample_numpy(self, pts: np.ndarray) -> np.ndarray:
        if pts.size == 0:
            return pts

        v = max(self.voxel, 1e-6)
        off = self.voxel_offset_m

        # 정수 격자 인덱스 (음수 좌표 방지용 양수 오프셋)
        idx = np.floor((pts + off) / v).astype(np.int32)  # (N,3)
        # 각 행을 고정 길이 바이트로 뷰 변환해 고유 셀 탐색
        keys = idx.view(dtype=np.dtype((np.void, idx.dtype.itemsize * idx.shape[1]))).ravel()
        uniq_keys, inv = np.unique(keys, return_inverse=True)

        if self.voxel_method == "first":
            # 각 셀의 첫 번째 포인트 선택 (가장 빠름)
            first_idx = np.zeros(uniq_keys.size, dtype=np.int64)
            # 첫 등장 인덱스
            np.minimum.at(first_idx, inv, np.arange(inv.size, dtype=np.int64))
            return pts[first_idx].astype(np.float32)

        # 기본: centroid (셀 평균)
        out = np.zeros((uniq_keys.size, 3), dtype=np.float64)
        np.add.at(out, inv, pts)
        counts = np.bincount(inv)
        out /= counts[:, None]
        return out.astype(np.float32)

    def _process_latest(self):
        t0 = time.perf_counter()

        with self._lock:
            msg = self._latest_msg
            self._latest_msg = None
        if msg is None:
            return

        # 프레임 신선도 체크
        now_ns = self.get_clock().now().nanoseconds
        msg_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        age_ms = (now_ns - msg_ns) / 1e6
        if age_ms > self.max_age_ms:
            return

        # 필터 비활성화면 원본 패스스루
        if not self.enable_filter:
            if self.pub_rel: self.pub_rel.publish(msg)
            if self.pub_be:  self.pub_be.publish(msg)
            return

        # 1) PointCloud2 -> NumPy (x,y,z)
        try:
            points_raw = pc2.read_points_numpy(msg, field_names=("x", "y", "z"))
        except Exception as e:
            self.get_logger().warn(f"[filter] read_points_numpy failed: {e}")
            return
        if points_raw.size == 0:
            return

        # 2) ROI/FOV 마스킹
        x = points_raw[:, 0]
        y = points_raw[:, 1]
        z = points_raw[:, 2]

        z_mask   = (z > self.z_min) & (z < self.z_max)
        y_mask   = (y > self.y_min) & (y < self.y_max)
        fov_mask = (z > 0.0) & (np.abs(x) < (z * self.fov_half_tan))
        mask = z_mask & y_mask & fov_mask
        if not np.any(mask):
            return

        filtered = points_raw[mask]

        # 3) 보xel 다운샘플(NumPy)
        out_pts = self._voxel_downsample_numpy(filtered)
        if out_pts.size == 0:
            return

        # 4) NumPy -> PointCloud2
        out_msg = pc2.create_cloud_xyz32(msg.header, out_pts)
        if self.pub_rel: self.pub_rel.publish(out_msg)
        if self.pub_be:  self.pub_be.publish(out_msg)

        # 타이밍 가드: 30Hz 예산 초과 경고(1초에 한 번)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if dt_ms > self.warn_budget_ms:
            self.get_logger().warn(
                f"[filter] {dt_ms:.1f} ms > budget {self.warn_budget_ms:.1f} ms "
                f"(in={points_raw.shape[0]}, after_mask={filtered.shape[0]}, out={out_pts.shape[0]}, "
                f"voxel={self.voxel}, method={self.voxel_method})",
                throttle_duration_sec=1.0
            )


def main(args=None):
    rclpy.init(args=args)
    node = LatestOnlyFilter()
    try:
        from rclpy.executors import MultiThreadedExecutor
        exec = MultiThreadedExecutor(num_threads=2)
        exec.add_node(node)
        exec.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
