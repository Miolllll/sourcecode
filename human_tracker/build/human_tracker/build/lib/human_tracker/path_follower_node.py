#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist, PointStamped
import numpy as np
import tf2_ros
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point
from rclpy.time import Time
import math

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

class PurePursuitController(Node):
    def __init__(self):
        super().__init__('pure_pursuit_controller')

        # --- Parameters (기존) ---
        self.declare_parameter('path_topic', '/global_plan')
        self.declare_parameter('user_topic', '/user_tracking')
        self.declare_parameter('cmd_topic',  '/cmd_vel')

        self.declare_parameter('lookahead_distance', 0.8)     # L (기본값)
        self.declare_parameter('target_distance',    0.9)
        self.declare_parameter('min_linear_speed',   0.13)  # default: 0.10
        self.declare_parameter('max_linear_speed',   0.30)
        self.declare_parameter('max_angular_speed',  0.50)

        self.declare_parameter('deadband_eps',       0.02)
        self.declare_parameter('restart_hysteresis', 0.15)
        self.declare_parameter('a_max',              0.8)
        self.declare_parameter('use_curvature_limit', True)
        self.declare_parameter('kappa_speed_gain',     1.0)

        self.declare_parameter('assume_robot_at_origin', True)
        self.declare_parameter('assume_heading_plus_x',  True)

        # --- Parameters (추가) ---
        # 동적 Lookahead: L = clamp(L0 + kv*|v|, Lmin, Lmax)
        self.declare_parameter('use_vel_scaled_lookahead', True)
        self.declare_parameter('L0',   1.5)
        self.declare_parameter('kv',   1.0)
        self.declare_parameter('Lmin', 0.8)
        self.declare_parameter('Lmax', 2.3)  # default: 2.3

        # α(헤딩오차) 기반 PD(+I) 댐핑
        self.declare_parameter('alpha_filter', 0.25)   # 0~1, 작을수록 부드럽게
        self.declare_parameter('k_alpha_d',   0.6)     # D 게인(진동 억제 핵심)
        self.declare_parameter('k_alpha_i',   0.0)     # I 게인(기본 0으로 시작)
        self.declare_parameter('i_clamp',     0.5)     # 적분 한계(라디안)

        # 각속도 후처리
        self.declare_parameter('w_slew_rate', 1.0)     # [rad/s^2]
        self.declare_parameter('w_deadband',  0.02)    # [rad/s]
        self.declare_parameter('w_limit',     0.5)     # [rad/s] (wmax와 함께 적용)

        # --- Turn-aware lookahead (NEW) ---
        self.declare_parameter('reduce_turn_lookahead', True)
        self.declare_parameter('alpha_on',   0.2)  # [rad] 여기서부터 "회전 시작"
        self.declare_parameter('alpha_full', 0.6)  # [rad] 이 이상은 "완전 회전"
        self.declare_parameter('k_alpha_L',  0.8)  # [m]  최대 lookahead 감쇠량
        self.declare_parameter('alpha_power', 2.0) # s^p 의 p (1~2 추천)

        # --- Load params ---
        path_topic = self.get_parameter('path_topic').value
        user_topic = self.get_parameter('user_topic').value
        cmd_topic  = self.get_parameter('cmd_topic').value

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

        self.reduce_turn_L = bool(self.get_parameter('reduce_turn_lookahead').value)
        self.alpha_on   = float(self.get_parameter('alpha_on').value)
        self.alpha_full = float(self.get_parameter('alpha_full').value)
        self.k_alpha_L  = float(self.get_parameter('k_alpha_L').value)
        self.alpha_pow  = float(self.get_parameter('alpha_power').value)


        # TF (user -> path_frame 변환)
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- State ---
        self.current_path_pts = []     # [(x,y), ...] in path_frame
        self.path_frame = None
        self.user_point_in_path = None # (x,y) in path_frame

        self.v_prev = 0.0
        self.latched_stop = False

        # α-PD 상태
        self.alpha_prev = 0.0
        self.alpha_filt = 0.0
        self.int_alpha  = 0.0

        # w rate-limit 상태
        self.w_prev = 0.0

        # --- QoS / I/O ---
        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.path_sub = self.create_subscription(Path, path_topic, self.on_path, 10)
        self.user_sub = self.create_subscription(PointStamped, user_topic, self.on_user, 10)
        self.cmd_pub  = self.create_publisher(Twist, cmd_topic, 10)

        # Control loop: 20 Hz
        self.dt = 0.05
        self.timer = self.create_timer(self.dt, self.control_step)

        self.get_logger().info('PurePursuitController (with PD damping) started.')

    # ----------------- Callbacks -----------------
    def on_path(self, msg: Path):
        self.path_frame = msg.header.frame_id if msg.header.frame_id else 'map'
        self.current_path_pts = [(ps.pose.position.x, ps.pose.position.y) for ps in msg.poses]

    def on_user(self, msg: PointStamped):
        if self.path_frame is None:
            self.user_point_in_path = (msg.point.x, msg.point.y)
            return
        try:
            trans = self.tf_buffer.lookup_transform(
                self.path_frame,
                msg.header.frame_id if msg.header.frame_id else self.path_frame,
                Time()
            )
            pt_in_path = do_transform_point(msg, trans)
            self.user_point_in_path = (pt_in_path.point.x, pt_in_path.point.y)
        except Exception:
            self.user_point_in_path = (msg.point.x, msg.point.y)

    # ----------------- Control -----------------
    def control_step(self):
        if not self.current_path_pts:
            self.publish_cmd(0.0, 0.0)
            return

        # 로봇의 위치/헤딩 가정(현재 구조 유지: 경로가 로봇 기준 프레임이어야 정확)
        robot_xy = np.array([0.0, 0.0])
        robot_heading = 0.0  # +x

        # base L (기존 동적/고정 로직 유지)
        if self.use_vel_scaled_L:
            base_L = clamp(self.L0 + self.kv * abs(self.v_prev), self.Lmin, self.Lmax)
        else:
            base_L = clamp(self.L_base, self.Lmin, self.Lmax)

        L_eff = base_L
        if self.reduce_turn_L and self.alpha_full > 1e-6:
            a = abs(self.alpha_filt)  # 이전 틱에서 필터된 헤딩오차
            # smoothstep-like 스케일 0..1
            if a <= self.alpha_on:
                s = 0.0
            elif a >= self.alpha_full:
                s = 1.0
            else:
                t = (a - self.alpha_on) / (self.alpha_full - self.alpha_on)
                # 부드럽게 (3t^2 - 2t^3) 쓰거나, 간단히 t^p 사용
                s = t ** self.alpha_pow   # 간단/안정: t^p

            L_eff = clamp(base_L - self.k_alpha_L * s, self.Lmin, self.Lmax)

        # Lookahead point
        look_pt = self.find_lookahead_point(robot_xy, self.current_path_pts, L_eff)
        if look_pt is None:
            self.publish_cmd(0.0, 0.0)
            return

        # Pure Pursuit 기하
        dx, dy = look_pt[0] - robot_xy[0], look_pt[1] - robot_xy[1]
        ld = float(np.hypot(dx, dy))
        if ld < 1e-6:
            self.publish_cmd(0.0, 0.0)
            return

        alpha = math.atan2(dy, dx) - robot_heading
        alpha = (alpha + math.pi) % (2*math.pi) - math.pi

        # α 저역통과
        a = clamp(self.alpha_a, 0.0, 1.0)
        self.alpha_filt = (1.0 - a) * self.alpha_filt + a * alpha

        # v_target: 기존 정책
        v_target = self.compute_v(robot_xy)

        # 곡률 기반 v 상한
        if self.use_kappa_limit and v_target > 0.0:
            kappa_pp = 2.0 * math.sin(self.alpha_filt) / (ld + 1e-9)
            v_curve = self.vmax / (1.0 + self.kappa_gain * abs(kappa_pp))
            v_target = float(min(v_target, v_curve))

        # 가속도 제한
        dv = clamp(v_target - self.v_prev, -self.a_max * self.dt, self.a_max * self.dt)
        v_cmd = self.v_prev + dv
        if v_cmd < 1e-3:
            v_cmd = 0.0
        self.v_prev = float(v_cmd)

        # --- PP 기본 각속도 ---
        w_pp = 2.0 * v_cmd * math.sin(self.alpha_filt) / (ld + 1e-9)

        # --- α-PD(+I) 댐핑 ---
        d_alpha = (self.alpha_filt - self.alpha_prev) / max(self.dt, 1e-6)
        self.alpha_prev = self.alpha_filt

        # 적분(기본 off; 쓰면 아주 작게, 포화 포함)
        self.int_alpha += self.alpha_filt * self.dt
        self.int_alpha = clamp(self.int_alpha, -self.i_clamp, self.i_clamp)

        w_pid = self.kd * d_alpha + self.ki * self.int_alpha   # (P는 PP가 맡음)
        w_cmd = w_pp + w_pid

        # 각속도 후처리: 데드밴드 -> 슬루 -> 포화
        if abs(w_cmd) < self.w_db:
            w_cmd = 0.0
        max_dw = self.w_slew * self.dt
        w_cmd = clamp(w_cmd, self.w_prev - max_dw, self.w_prev + max_dw)
        w_cmd = clamp(w_cmd, -min(self.wmax, self.w_lim2), min(self.wmax, self.w_lim2))
        self.w_prev = w_cmd

        self.publish_cmd(float(v_cmd), float(w_cmd))

    # ----------------- Helpers -----------------
    def compute_v(self, robot_xy: np.ndarray) -> float:
        if self.user_point_in_path is None:
            if self.current_path_pts:
                ux, uy = self.current_path_pts[-1]
            else:
                return 0.0
        else:
            ux, uy = self.user_point_in_path

        D = float(np.hypot(ux - robot_xy[0], uy - robot_xy[1]))

        if D < self.Dt:
            self.latched_stop = True
            return 0.0
        if self.latched_stop and D < (self.Dt + self.restart_hyst):
            return 0.0
        self.latched_stop = False

        if abs(D - self.Dt) <= self.deadband_eps:
            return float(np.clip(self.v_prev, self.vmin, self.vmax))

        scale = min(max((D - self.Dt) / max(self.Dt, 1e-6), 0.0), 1.0)
        v = self.vmin + (self.vmax - self.vmin) * scale
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
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
