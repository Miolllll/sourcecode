#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, OccupancyGrid, MapMetaData
from geometry_msgs.msg import Twist, PointStamped, TransformStamped
from sensor_msgs.msg import Imu  # ★ IMU
import numpy as np
import tf2_ros
from tf2_ros import Buffer, TransformListener, StaticTransformBroadcaster
from tf2_geometry_msgs import do_transform_point
from rclpy.time import Time
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy  # ★ IMU QoS
import math
from typing import Optional, Tuple

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def yaw_to_quat(yaw: float):
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))

class PurePursuitController(Node):
    def __init__(self):
        super().__init__('pure_pursuit_controller')

        # --- Topics & Frames ---
        self.declare_parameter('path_topic', '/global_plan')
        self.declare_parameter('user_topic', '/user_tracking')
        self.declare_parameter('cmd_topic',  '/cmd_vel')
        self.declare_parameter('base_frame', 'rear_axle')
        self.declare_parameter('distance_frame', 'camera_base')

        # --- Auto static TF (camera_base -> rear_axle) ---
        self.declare_parameter('publish_rear_axle_tf', True)
        self.declare_parameter('camera_frame', 'camera_base')
        self.declare_parameter('rear_axle_offset_x', -0.87)
        self.declare_parameter('rear_axle_offset_y',  0.0)
        self.declare_parameter('rear_axle_offset_z',  0.0)
        self.declare_parameter('rear_axle_yaw_rpy',   0.0)

        # --- Pure Pursuit / Stop / Limits ---
        self.declare_parameter('lookahead_distance', 0.8)
        self.declare_parameter('target_distance',    1.1)
        self.declare_parameter('min_linear_speed',   0.15)
        self.declare_parameter('max_linear_speed',   0.70)
        self.declare_parameter('max_angular_speed',  1.50)
        self.declare_parameter('deadband_eps',       0.02)
        self.declare_parameter('restart_hysteresis', 0.15)
        self.declare_parameter('a_max',              0.8)
        self.declare_parameter('use_curvature_limit', True)
        self.declare_parameter('kappa_speed_gain',     1.0)

        # --- Geometry assumptions ---
        self.declare_parameter('assume_robot_at_origin', True)
        self.declare_parameter('assume_heading_plus_x',  True)

        # --- Dynamic lookahead ---
        self.declare_parameter('use_vel_scaled_lookahead', True)
        self.declare_parameter('L0',   1.10)  ##낮추기
        self.declare_parameter('kv',   0.9)  ## 낮
        self.declare_parameter('Lmin', 0.7)  ## 낮
        self.declare_parameter('Lmax', 2.3)

        # --- Heading filter / PID on alpha ---
        self.declare_parameter('alpha_filter', 0.35)
        self.declare_parameter('k_alpha_d',   0.6)
        self.declare_parameter('k_alpha_i',   0.0)
        self.declare_parameter('i_clamp',     0.5)

        # --- Angular slew/limits ---
        self.declare_parameter('w_slew_rate', 1.0)
        self.declare_parameter('w_deadband',  0.02)
        self.declare_parameter('w_limit',     2.00)  ## 높

        # --- Hard stop window ---
        self.declare_parameter('use_hard_stop', True)
        self.declare_parameter('stop_distance', 1.1)
        self.declare_parameter('slow_distance', 1.3)
        self.declare_parameter('stop_release_hysteresis', 0.1)

        # --- Soft start ---
        self.declare_parameter('use_soft_start', True)
        self.declare_parameter('soft_a',        0.35)
        self.declare_parameter('vmin_start',    0.0)
        self.declare_parameter('soft_min_time', 1.0)
        self.declare_parameter('soft_w_scale',  True)

        # --- Obstacle Stop (map) ---
        self.declare_parameter('obstacle_stop_enable', True)
        self.declare_parameter('obstacle_map_topic', '/planner/persistent_inflated_map')
        self.declare_parameter('obstacle_occ_threshold', 50)
        self.declare_parameter('obstacle_stop_distance_m', 0.8)
        self.declare_parameter('obstacle_stop_width_m',   0.90)
        self.declare_parameter('obstacle_stop_margin_m',  0.10)
        self.declare_parameter('obstacle_resume_hyst_m',  0.20)

        # --- NEW: Stanley CTE term ---
        self.declare_parameter('use_cte_term', True)
        self.declare_parameter('k_cte',        1.0)
        self.declare_parameter('stanley_eps',  0.2)

        # --- NEW: IMU yaw-rate damping (anti-S wobble) ---
        self.declare_parameter('use_imu_yaw_damping', True)
        self.declare_parameter('imu_topic', '/k4a/imu')
        self.declare_parameter('k_gyro', -1.15)  ## +0.15~0.30             # ↑/↓ 튜닝 (0.15~0.30)
        self.declare_parameter('gyro_lpf_tau', 0.05)       # [s] 0이면 LPF 미사용
        self.declare_parameter('use_gyro_bias_est', True)  # 정지 시 바이어스 적분
        self.declare_parameter('gyro_bias_alpha', 0.001)   # 매우 느리게

        # --- Read params ---
        path_topic      = self.get_parameter('path_topic').value
        user_topic      = self.get_parameter('user_topic').value
        cmd_topic       = self.get_parameter('cmd_topic').value
        self.base_frame = self.get_parameter('base_frame').value
        self.distance_frame = self.get_parameter('distance_frame').value

        self.publish_tf  = bool(self.get_parameter('publish_rear_axle_tf').value)
        self.camera_frame = self.get_parameter('camera_frame').value
        self.offset_x = float(self.get_parameter('rear_axle_offset_x').value)
        self.offset_y = float(self.get_parameter('rear_axle_offset_y').value)
        self.offset_z = float(self.get_parameter('rear_axle_offset_z').value)
        self.offset_yaw = float(self.get_parameter('rear_axle_yaw_rpy').value)

        self.L_base = float(self.get_parameter('lookahead_distance').value)
        self.Dt   = float(self.get_parameter('target_distance').value)
        self.vmin = float(self.get_parameter('min_linear_speed').value)
        self.vmax = float(self.get_parameter('max_linear_speed').value)
        self.wmax = float(self.get_parameter('max_angular_speed').value)
        self.deadband_eps = float(self.get_parameter('deadband_eps').value)
        self.restart_hyst = float(self.get_parameter('restart_hysteresis').value)
        self.a_max        = float(self.get_parameter('a_max').value)
        self.use_kappa_limit = bool(self.get_parameter('use_curvature_limit').value)
        self.kappa_gain      = float(self.get_parameter('kappa_speed_gain').value)
        self.use_vel_scaled_L = bool(self.get_parameter('use_vel_scaled_lookahead').value)
        self.L0   = float(self.get_parameter('L0').value)
        self.kv   = float(self.get_parameter('kv').value)
        self.Lmin = float(self.get_parameter('Lmin').value)
        self.Lmax = float(self.get_parameter('Lmax').value)
        self.alpha_a = float(self.get_parameter('alpha_filter').value)
        self.kd = float(self.get_parameter('k_alpha_d').value)
        self.ki = float(self.get_parameter('k_alpha_i').value)
        self.i_clamp = float(self.get_parameter('i_clamp').value)
        self.w_slew = float(self.get_parameter('w_slew_rate').value)
        self.w_db   = float(self.get_parameter('w_deadband').value)
        self.w_lim2 = float(self.get_parameter('w_limit').value)
        self.assume_origin   = bool(self.get_parameter('assume_robot_at_origin').value)
        self.assume_heading  = bool(self.get_parameter('assume_heading_plus_x').value)
        self.use_hard_stop   = bool(self.get_parameter('use_hard_stop').value)
        self.stop_distance   = float(self.get_parameter('stop_distance').value)
        self.slow_distance   = float(self.get_parameter('slow_distance').value)
        self.stop_release    = float(self.get_parameter('stop_release_hysteresis').value)
        self.use_soft_start  = bool(self.get_parameter('use_soft_start').value)
        self.soft_a          = float(self.get_parameter('soft_a').value)
        self.vmin_start      = float(self.get_parameter('vmin_start').value)
        self.soft_min_time   = float(self.get_parameter('soft_min_time').value)
        self.soft_w_scale    = bool(self.get_parameter('soft_w_scale').value)

        # Obstacle Stop
        self.obstacle_stop_enable = bool(self.get_parameter('obstacle_stop_enable').value)
        self.obstacle_map_topic   = self.get_parameter('obstacle_map_topic').value
        self.obstacle_occ_th      = int(self.get_parameter('obstacle_occ_threshold').value)
        self.obs_stop_dist        = float(self.get_parameter('obstacle_stop_distance_m').value)
        self.obs_stop_width       = float(self.get_parameter('obstacle_stop_width_m').value)
        self.obs_stop_margin      = float(self.get_parameter('obstacle_stop_margin_m').value)
        self.obs_resume_hyst      = float(self.get_parameter('obstacle_resume_hyst_m').value)

        # NEW: CTE params
        self.use_cte_term = bool(self.get_parameter('use_cte_term').value)
        self.k_cte        = float(self.get_parameter('k_cte').value)
        self.stanley_eps  = float(self.get_parameter('stanley_eps').value)

        # IMU params/state
        self.use_imu_yaw_damping = bool(self.get_parameter('use_imu_yaw_damping').value)
        self.imu_topic = self.get_parameter('imu_topic').value
        self.k_gyro    = float(self.get_parameter('k_gyro').value)
        self.gyro_lpf_tau = float(self.get_parameter('gyro_lpf_tau').value)
        self.use_gyro_bias_est = bool(self.get_parameter('use_gyro_bias_est').value)
        self.gyro_bias_alpha   = float(self.get_parameter('gyro_bias_alpha').value)

        # --- TF / State ---
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.static_broadcaster = None
        self._maybe_publish_static_tf()

        self.current_path_pts = []           # base_frame 좌표
        self.path_frame = None
        self.user_point_local = None         # base_frame (추종)
        self.user_point_for_distance = None  # distance_frame (거리 규칙)

        self.v_prev = 0.0
        self.latched_stop = False
        self.hard_stopped = False
        self.alpha_prev = 0.0
        self.alpha_filt = 0.0
        self.int_alpha  = 0.0
        self.w_prev = 0.0
        self.move_time = 0.0

        # Obstacle map cache
        self.obs_grid: Optional[np.ndarray] = None
        self.obs_info: Optional[MapMetaData] = None
        self.obs_frame: Optional[str] = None
        self.obstacle_blocking = False

        # ★ IMU 상태
        self.gyro_z_raw = 0.0
        self.gyro_z_filt = 0.0
        self.gyro_bias = 0.0

        # --- ROS IO ---
        self.path_sub = self.create_subscription(Path, path_topic, self.on_path, 10)
        self.user_sub = self.create_subscription(PointStamped, user_topic, self.on_user, 10)
        if self.obstacle_stop_enable:
            self.obs_sub  = self.create_subscription(OccupancyGrid, self.obstacle_map_topic, self.on_obstacle_map, 10)

        # 타이밍은 IMU 필터에서 쓰니 선행 설정
        self.dt = 0.05

        # ★ IMU QoS (Azure Kinect는 Best Effort가 흔함)
        if self.use_imu_yaw_damping:
            imu_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=50
            )
            self.imu_sub = self.create_subscription(Imu, self.imu_topic, self.on_imu, imu_qos)

        self.cmd_pub  = self.create_publisher(Twist, cmd_topic, 10)
        self.timer = self.create_timer(self.dt, self.control_step)

        self.get_logger().info(f"[PurePursuit] base_frame='{self.base_frame}', distance_frame='{self.distance_frame}', "
                               f"ObstacleStop={self.obstacle_stop_enable} | IMU={self.use_imu_yaw_damping}")

    # --- optional static TF publish ---
    def _maybe_publish_static_tf(self):
        if not self.publish_tf:
            return
        if self.base_frame == self.camera_frame:
            self.get_logger().warn("[PurePursuit] base_frame == camera_frame; skip publishing rear_axle TF.")
            return
        self.static_broadcaster = StaticTransformBroadcaster(self)
        ts = TransformStamped()
        ts.header.stamp = self.get_clock().now().to_msg()
        ts.header.frame_id = self.camera_frame
        ts.child_frame_id  = self.base_frame
        ts.transform.translation.x = float(self.offset_x)
        ts.transform.translation.y = float(self.offset_y)
        ts.transform.translation.z = float(self.offset_z)
        qx, qy, qz, qw = yaw_to_quat(self.offset_yaw)
        ts.transform.rotation.x = qx
        ts.transform.rotation.y = qy
        ts.transform.rotation.z = qz
        ts.transform.rotation.w = qw
        self.static_broadcaster.sendTransform(ts)
        self.get_logger().info(f"[PurePursuit] Published static TF: {self.camera_frame} -> {self.base_frame} "
                               f"(dx={self.offset_x:.3f}, dy={self.offset_y:.3f}, dz={self.offset_z:.3f}, yaw={self.offset_yaw:.3f})")

    # ===== Callbacks =====
    def on_path(self, msg: Path):
        self.path_frame = msg.header.frame_id if msg.header.frame_id else 'map'
        self.current_path_pts = []
        try:
            trans = self.tf_buffer.lookup_transform(self.base_frame, self.path_frame, Time())
            for ps in msg.poses:
                pt = PointStamped()
                pt.header = msg.header
                pt.point.x = ps.pose.position.x
                pt.point.y = ps.pose.position.y
                pt.point.z = 0.0
                pt_local = do_transform_point(pt, trans)
                self.current_path_pts.append((pt_local.point.x, pt_local.point.y))
        except Exception as e:
            self.get_logger().warn(f"[TF] path->'{self.base_frame}' transform failed: {e}. Using raw path points.")
            self.current_path_pts = [(ps.pose.position.x, ps.pose.position.y) for ps in msg.poses]

    def on_user(self, msg: PointStamped):
        # 1) 추종용: base_frame
        try:
            if msg.header.frame_id == self.base_frame:
                self.user_point_local = (msg.point.x, msg.point.y)
            else:
                trans_b = self.tf_buffer.lookup_transform(self.base_frame, msg.header.frame_id, Time())
                pt_b = do_transform_point(msg, trans_b)
                self.user_point_local = (pt_b.point.x, pt_b.point.y)
        except Exception as e:
            self.get_logger().warn(f"[TF] user->'{self.base_frame}' transform failed: {e}. Using raw for local.")
            self.user_point_local = (msg.point.x, msg.point.y)

        # 2) 거리규칙용: distance_frame
        try:
            if self.distance_frame == self.base_frame and self.user_point_local is not None:
                self.user_point_for_distance = self.user_point_local
            elif msg.header.frame_id == self.distance_frame:
                self.user_point_for_distance = (msg.point.x, msg.point.y)
            else:
                trans_d = self.tf_buffer.lookup_transform(self.distance_frame, msg.header.frame_id, Time())
                pt_d = do_transform_point(msg, trans_d)
                self.user_point_for_distance = (pt_d.point.x, pt_d.point.y)
        except Exception as e:
            self.get_logger().warn(f"[TF] user->'{self.distance_frame}' transform failed: {e}. Fallback to base_frame distance.")
            self.user_point_for_distance = self.user_point_local

    def on_obstacle_map(self, msg: OccupancyGrid):
        self.obs_frame = msg.header.frame_id if msg.header.frame_id else 'map'
        self.obs_info = msg.info
        data = np.array(msg.data, dtype=np.int16).reshape((msg.info.height, msg.info.width))
        self.obs_grid = data

    # ★ IMU 콜백: 1차 LPF + (정지 시) 바이어스 추정
    def on_imu(self, msg: Imu):
        z = float(msg.angular_velocity.z)  # rad/s
        # LPF
        if self.gyro_lpf_tau > 1e-6:
            alpha = self.dt / (self.gyro_lpf_tau + self.dt)
            self.gyro_z_filt = self.gyro_z_filt + alpha * (z - self.gyro_z_filt)
        else:
            self.gyro_z_filt = z
        self.gyro_z_raw = z
        # 정지 상태에서 매우 천천히 바이어스 적분
        if self.use_gyro_bias_est:
            if abs(self.v_prev) < 0.05 and abs(self.w_prev) < 0.05:
                a = self.gyro_bias_alpha
                self.gyro_bias = (1.0 - a) * self.gyro_bias + a * self.gyro_z_filt

    # ===== Geometry helper: signed CTE =====
    def signed_cte(self, path_pts) -> float:
        """base_frame에서 (0,0)을 경로 세그먼트에 투영해 부호 있는 횡오차를 계산"""
        if not path_pts or len(path_pts) < 2:
            return 0.0
        P = np.array(path_pts, dtype=float)
        best_d2 = 1e18; best_cte = 0.0
        for i in range(len(P)-1):
            p0 = P[i]; p1 = P[i+1]
            s = p1 - p0
            L = float(np.linalg.norm(s))
            if L < 1e-9:
                continue
            t = float(np.dot(-p0, s) / (L*L))
            t = clamp(t, 0.0, 1.0)
            q = p0 + t * s
            r_q = -q
            d2 = float(np.dot(r_q, r_q))
            if d2 < best_d2:
                cross_z = s[0]*r_q[1] - s[1]*r_q[0]
                best_cte = cross_z / (L + 1e-9)
                best_d2 = d2
        return float(best_cte)

    # ===== Control Loop =====
    def control_step(self):
        if not self.current_path_pts:
            self._reset_movement_timer()
            self.publish_cmd(0.0, 0.0)
            return

        robot_xy = np.array([0.0, 0.0])  # base_frame
        robot_heading = 0.0

        # ---- 사람 거리 규칙 ----
        if self.user_point_for_distance is None:
            ux_d, uy_d = self.current_path_pts[-1] if self.current_path_pts else (1e9, 1e9)
        else:
            ux_d, uy_d = self.user_point_for_distance
        D = float(np.hypot(ux_d, uy_d))

        if self.use_hard_stop:
            if self.hard_stopped:
                if D < (self.stop_distance + self.stop_release):
                    self._reset_movement_timer(); self.publish_cmd(0.0, 0.0); return
                else:
                    self.hard_stopped = False
            if D <= self.stop_distance:
                self.hard_stopped = True
                self._reset_movement_timer(); self.publish_cmd(0.0, 0.0); return

        if D <= self.Dt:
            self.latched_stop = True
        if self.latched_stop:
            if D < (self.Dt + self.restart_hyst):
                self._reset_movement_timer(); self.publish_cmd(0.0, 0.0); return
            else:
                self.latched_stop = False

        # ---- 장애물 급정지 ----
        if self.obstacle_stop_enable and self._front_zone_blocked():
            self.obstacle_blocking = True
            self._reset_movement_timer(); self.publish_cmd(0.0, 0.0); return
        else:
            if self.obstacle_blocking:
                if self._front_zone_blocked(clearance=self.obs_resume_hyst):
                    self._reset_movement_timer(); self.publish_cmd(0.0, 0.0); return
                else:
                    self.obstacle_blocking = False

        # ---- lookahead ----
        L_eff = clamp(self.L0 + self.kv * abs(self.v_prev), self.Lmin, self.Lmax) \
                if self.use_vel_scaled_L else self.L_base

        look_pt = self.find_lookahead_point(robot_xy, self.current_path_pts, L_eff)
        if look_pt is None:
            self._reset_movement_timer(); self.publish_cmd(0.0, 0.0); return

        dx, dy = look_pt[0] - robot_xy[0], look_pt[1] - robot_xy[1]
        ld = float(np.hypot(dx, dy))
        if ld < 1e-6:
            self._reset_movement_timer(); self.publish_cmd(0.0, 0.0); return

        alpha = (math.atan2(dy, dx) - robot_heading + math.pi) % (2*math.pi) - math.pi

        a = clamp(self.alpha_a, 0.0, 1.0)
        self.alpha_filt = (1.0 - a) * self.alpha_filt + a * alpha

        self.move_time += self.dt
        v_target = self.compute_v(D, self.move_time)

        if self.use_soft_start:
            v_soft_cap = max(0.0, self.soft_a * self.move_time)
            v_target = min(v_target, v_soft_cap)

        if self.use_kappa_limit and v_target > 0.0:
            kappa_pp = 2.0 * math.sin(self.alpha_filt) / (ld + 1e-9)
            v_curve = self.vmax / (1.0 + self.kappa_gain * abs(kappa_pp))
            v_target = float(min(v_target, v_curve))

        dv = clamp(v_target - self.v_prev, -self.a_max * self.dt, self.a_max * self.dt)
        v_cmd = self.v_prev + dv
        if v_cmd < 1e-3:
            v_cmd = 0.0

        if self.use_hard_stop and (self.slow_distance > self.stop_distance) and (D < self.slow_distance):
            scale = clamp((D - self.stop_distance) / (self.slow_distance - self.stop_distance), 0.0, 1.0)
            v_cmd *= scale

        self.v_prev = float(v_cmd)

        # --- 회전각속도: PP + alpha-PD + (Stanley CTE) ---
        w_pp = 2.0 * v_cmd * math.sin(self.alpha_filt) / (ld + 1e-9)
        d_alpha = (self.alpha_filt - self.alpha_prev) / max(self.dt, 1e-6)
        self.alpha_prev = self.alpha_filt
        self.int_alpha = clamp(self.int_alpha + self.alpha_filt * self.dt, -self.i_clamp, self.i_clamp)
        w_pid = self.kd * d_alpha + self.ki * self.int_alpha

        w_cte = 0.0
        if self.use_cte_term:
            cte = self.signed_cte(self.current_path_pts)
            w_cte = math.atan2(self.k_cte * cte, abs(v_cmd) + self.stanley_eps)

        w_cmd = w_pp + w_pid + w_cte

        # ★ IMU 요레이트 감쇠: 실제 회전속도(gyro_z)로 흔들림 상쇄
        if self.use_imu_yaw_damping:
            gyro_used = self.gyro_z_filt - (self.gyro_bias if self.use_gyro_bias_est else 0.0)
            w_cmd -= self.k_gyro * gyro_used

        if self.use_soft_start and self.soft_w_scale:
            w_cmd *= clamp(self.move_time / max(self.soft_min_time, 1e-6), 0.0, 1.0)

        if abs(w_cmd) < self.w_db:
            w_cmd = 0.0
        max_dw = self.w_slew * self.dt
        w_cmd = clamp(w_cmd, self.w_prev - max_dw, self.w_prev + max_dw)
        w_cmd = clamp(w_cmd, -min(self.wmax, self.w_lim2), min(self.wmax, self.w_lim2))
        self.w_prev = w_cmd

        self.publish_cmd(float(v_cmd), float(w_cmd))

    # ===== Obstacle Stop helpers =====
    def on_grid_idx(self, info: MapMetaData, x_m: float, y_m: float) -> Optional[Tuple[int, int]]:
        i = int((x_m - info.origin.position.x) / info.resolution)
        j = int((y_m - info.origin.position.y) / info.resolution)
        if 0 <= i < info.width and 0 <= j < info.height:
            return (i, j)
        return None

    def _front_zone_blocked(self, clearance: float = 0.0) -> bool:
        if self.obs_grid is None or self.obs_info is None or self.obs_frame is None:
            return False
        res = float(self.obs_info.resolution)
        stop_dist = self.obs_stop_dist + clearance
        half_w = 0.5 * (self.obs_stop_width + 2.0 * self.obs_stop_margin)
        xs = np.arange(0.0, max(stop_dist, 0.0) + 1e-6, res)
        ys = np.arange(-half_w, half_w + 1e-6, res)
        try:
            trans = self.tf_buffer.lookup_transform(self.obs_frame, self.base_frame, Time())
        except Exception as e:
            self.get_logger().warn(f"[ObstacleStop] TF {self.base_frame}->{self.obs_frame} failed: {e}")
            return False
        for x in xs:
            for y in ys:
                pt = PointStamped()
                pt.header.frame_id = self.base_frame
                pt.header.stamp = self.get_clock().now().to_msg()
                pt.point.x = float(x); pt.point.y = float(y); pt.point.z = 0.0
                try:
                    pmap = do_transform_point(pt, trans)
                    idx = self.on_grid_idx(self.obs_info, pmap.point.x, pmap.point.y)
                    if idx is None: continue
                    i, j = idx
                    occ = int(self.obs_grid[j, i])
                    if occ >= self.obstacle_occ_th:
                        return True
                except Exception:
                    continue
        return False

    # ===== Helpers =====
    def _reset_movement_timer(self):
        self.move_time = 0.0
        self.v_prev = 0.0
        self.w_prev = 0.0

    def compute_v(self, D: float, move_time: float) -> float:
        vmin_eff = self.vmin if move_time >= self.soft_min_time else self.vmin_start
        if abs(D - self.Dt) <= self.deadband_eps:
            return float(np.clip(self.v_prev, vmin_eff, self.vmax))
        scale = min(max((D - self.Dt) / max(self.Dt, 1e-6), 0.0), 1.0)
        v = vmin_eff + (self.vmax - vmin_eff) * scale
        return float(np.clip(v, 0.0, self.vmax))

    def find_lookahead_point(self, robot_xy: np.ndarray, path_pts, L: float):
        if not path_pts:
            return None
        last = robot_xy
        acc = 0.0
        for pt in path_pts:
            seg = np.array(pt) - last
            seg_d = float(np.linalg.norm(seg))
            if acc + seg_d >= L:
                t = (L - acc) / (seg_d + 1e-9)
                target = last + t * seg
                return (float(target[0]), float(target[1]))
            acc += seg_d
            last = np.array(pt, dtype=float)
        return (float(path_pts[-1][0]), float(path_pts[-1][1]))

    def publish_cmd(self, v: float, w: float):
        msg = Twist()
        msg.linear.x = v
        msg.angular.z = w
        self.cmd_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
