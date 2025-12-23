#!/usr/bin/env python3
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rcl_interfaces.msg import SetParametersResult

from geometry_msgs.msg import Twist
from std_msgs.msg import Int32


class CmdVelToRpm(Node):
    """
    /cmd_vel(geometry_msgs/Twist) -> /cmd_rpm_left(Int32), /cmd_rpm_right(Int32)

    변환식 (차동 구동):
      v_l = v  - w * (W/2)
      v_r = v  + w * (W/2)
      RPM  = (v / (2*pi*r)) * 60

    파라미터:
      - wheel_radius_m     : 바퀴 반지름 [m]
      - track_width_m      : 좌우 바퀴 간 트랙 폭 [m]
      - max_wheel_rpm      : 휠 RPM 한계 (클램프)
      - cmd_timeout_s      : cmd_vel 수신 타임아웃(초) -> 타임아웃 시 0rpm
      - output_hz          : 출력 주기(Hz)
      - cmd_vel_topic      : (/cmd_vel 기본)
      - left_topic         : (/cmd_rpm_left 기본)
      - right_topic        : (/cmd_rpm_right 기본)
      - rpm_scale          : 계산된 RPM에 추가 스케일(보정용, 기본 1.0)
      - invert_linear_x    : 전/후 주행 방향 뒤집기 (기본 False)
      - invert_angular_z   : 회전 방향 뒤집기 (기본 True; 기존 코드의 w=-w 유지)
      - swap_wheels        : 좌/우 바꿈 (배선/드라이브단 뒤집힘 대응)
      - invert_left_rpm    : 좌 휠 회전방향 반전
      - invert_right_rpm   : 우 휠 회전방향 반전

    주의:
      - 실제 모터 구동 방향 반전은 md_controller 파라미터(invert_left/right)로 처리 권장.
      - 필요 시 여기서도 신속히 뒤집을 수 있도록 파라미터 제공.
      - md_controller에서 wheel RPM * GearRatio = 모터 RPM 으로 변환해 보냄.
    """

    def __init__(self):
        super().__init__("cmdvel_to_rpm")

        # ---- 파라미터 선언 ----
        self.declare_parameter("wheel_radius_m", 0.135)
        self.declare_parameter("track_width_m",  0.900)
        self.declare_parameter("max_wheel_rpm",  1000)
        self.declare_parameter("cmd_timeout_s",  0.50)
        self.declare_parameter("output_hz",      30.0)
        self.declare_parameter("cmd_vel_topic",  "/cmd_vel")
        self.declare_parameter("left_topic",     "/cmd_rpm_left")
        self.declare_parameter("right_topic",    "/cmd_rpm_right")
        self.declare_parameter("rpm_scale",      1.0)

        # 유연한 매핑(전륜↔후륜/앞뒤/좌우/방향 반전)
        self.declare_parameter("invert_linear_x",  False)  # 전륜 > True
        # 기존 코드의 w=-w 동작을 기본값으로 유지
        self.declare_parameter("invert_angular_z", True)  # 전륜 > True
        self.declare_parameter("swap_wheels",      False)
        self.declare_parameter("invert_left_rpm",  False)
        self.declare_parameter("invert_right_rpm", False)

        # ---- 파라미터 로드 ----
        self.r = float(self.get_parameter("wheel_radius_m").value)
        self.W = float(self.get_parameter("track_width_m").value)
        self.max_rpm     = int(self.get_parameter("max_wheel_rpm").value)
        self.timeout_s   = float(self.get_parameter("cmd_timeout_s").value)
        self.output_hz   = float(self.get_parameter("output_hz").value)
        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        self.left_topic    = self.get_parameter("left_topic").value
        self.right_topic   = self.get_parameter("right_topic").value
        self.rpm_scale     = float(self.get_parameter("rpm_scale").value)

        self.invert_linear_x  = bool(self.get_parameter("invert_linear_x").value)
        self.invert_angular_z = bool(self.get_parameter("invert_angular_z").value)
        self.swap_wheels      = bool(self.get_parameter("swap_wheels").value)
        self.invert_left_rpm  = bool(self.get_parameter("invert_left_rpm").value)
        self.invert_right_rpm = bool(self.get_parameter("invert_right_rpm").value)

        # ---- Pub/Sub ----
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.pub_left  = self.create_publisher(Int32, self.left_topic, qos)
        self.pub_right = self.create_publisher(Int32, self.right_topic, qos)
        self.sub = self.create_subscription(Twist, self.cmd_vel_topic, self._on_cmd_vel, qos)

        # ---- 상태 ----
        self.last_cmd_time = 0.0
        self.last_v = 0.0       # m/s
        self.last_w = 0.0       # rad/s

        # ---- 타이머 ----
        period = 1.0 / max(1e-6, self.output_hz)
        self.timer = self.create_timer(period, self._tick)

        # 동적 파라미터 업데이트 허용
        self.add_on_set_parameters_callback(self._on_param_change)

        self.get_logger().info(
            f"[cmdvel_to_rpm] r={self.r:.3f} m, W={self.W:.3f} m, "
            f"max_rpm={self.max_rpm}, timeout={self.timeout_s}s, out_hz={self.output_hz}"
        )
        self.get_logger().info(
            f"[cmdvel_to_rpm] cmd_vel='{self.cmd_vel_topic}', left='{self.left_topic}', right='{self.right_topic}', rpm_scale={self.rpm_scale}"
        )
        self.get_logger().info(
            f"[cmdvel_to_rpm] invert_linear_x={self.invert_linear_x}, invert_angular_z={self.invert_angular_z}, "
            f"swap_wheels={self.swap_wheels}, invL={self.invert_left_rpm}, invR={self.invert_right_rpm}"
        )

    # ----------------- 콜백 -----------------
    def _on_param_change(self, params):
        for p in params:
            if p.name == "invert_linear_x":
                self.invert_linear_x = bool(p.value)
            elif p.name == "invert_angular_z":
                self.invert_angular_z = bool(p.value)
            elif p.name == "swap_wheels":
                self.swap_wheels = bool(p.value)
            elif p.name == "invert_left_rpm":
                self.invert_left_rpm = bool(p.value)
            elif p.name == "invert_right_rpm":
                self.invert_right_rpm = bool(p.value)
            elif p.name == "wheel_radius_m":
                self.r = float(p.value)
            elif p.name == "track_width_m":
                self.W = float(p.value)
            elif p.name == "max_wheel_rpm":
                self.max_rpm = int(p.value)
            elif p.name == "cmd_timeout_s":
                self.timeout_s = float(p.value)
            elif p.name == "rpm_scale":
                self.rpm_scale = float(p.value)
        return SetParametersResult(successful=True)

    def _on_cmd_vel(self, msg: Twist):
        self.last_v = float(msg.linear.x)    # m/s
        self.last_w = float(msg.angular.z)   # rad/s
        self.last_cmd_time = time.time()

    # ----------------- 로직 -----------------
    def _tick(self):
        now = time.time()
        if now - self.last_cmd_time <= self.timeout_s:
            v = self.last_v
            w = self.last_w
        else:
            v = 0.0
            w = 0.0

        # 선택적 방향 반전
        if self.invert_linear_x:
            v = -v
        if self.invert_angular_z:
            w = -w

        # 차동 구동: 휠 선속도
        v_l = v - w * (self.W * 0.5)
        v_r = v + w * (self.W * 0.5)

        # 선속도 -> 휠 RPM
        two_pi_r = 2.0 * math.pi * self.r
        if two_pi_r <= 1e-9:
            self.get_logger().error("wheel_radius_m가 0 또는 너무 작습니다.")
            return

        rpm_l = (v_l / two_pi_r) * 60.0
        rpm_r = (v_r / two_pi_r) * 60.0

        # 좌/우 바꿈(필요 시)
        if self.swap_wheels:
            rpm_l, rpm_r = rpm_r, rpm_l

        # 보정 스케일
        rpm_l *= self.rpm_scale
        rpm_r *= self.rpm_scale

        # 개별 휠 회전방향 반전(필요 시)
        if self.invert_left_rpm:
            rpm_l = -rpm_l
        if self.invert_right_rpm:
            rpm_r = -rpm_r

        # 정수/클램프
        rpm_l = int(max(-self.max_rpm, min(self.max_rpm, round(rpm_l))))
        rpm_r = int(max(-self.max_rpm, min(self.max_rpm, round(rpm_r))))

        # 퍼블리시
        self.pub_left.publish(Int32(data=rpm_l))
        self.pub_right.publish(Int32(data=rpm_r))

# ----------------- 엔트리 -----------------
def main(args=None):
    rclpy.init(args=args)
    node = CmdVelToRpm()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
