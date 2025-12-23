#!/usr/bin/env python3
import math, time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

class ScriptRunner(Node):
    def __init__(self):
        super().__init__('md_script_runner')

        # ===== 기본 제어 파라미터 =====
        self.declare_parameter("wheel_radius_m", 0.135)
        self.declare_parameter("track_width_m",  0.900)
        self.declare_parameter("max_wheel_rpm",  100)   # 실내 안전
        self.declare_parameter("output_hz",      30.0)
        self.declare_parameter("left_topic",     "/cmd_rpm_left")
        self.declare_parameter("right_topic",    "/cmd_rpm_right")
        self.declare_parameter("rpm_scale",      1.0)

        # ✅ 바퀴 극성(앞이 +RPM이 되도록 보정)
        self.declare_parameter("left_polarity",  1)
        self.declare_parameter("right_polarity", 1)

        # ===== 시나리오 파라미터 =====
        # 1) 첫 직진
        self.declare_parameter('straight_rpm',     50)
        self.declare_parameter('straight_time_s',  2.0)

        # 2) 제자리 회전
        self.declare_parameter('turn_side',        'left')  # 'left' or 'right'
        self.declare_parameter('turn_rpm',         60)
        self.declare_parameter('turn_angle_deg',   90.0)
        
        # 회전 부족/과다 보정 (스케일, 바이어스)
        self.declare_parameter('turn_time_gain',   1.15)    # 10% 더 돌림(예: 마찰/슬립 보정)
        self.declare_parameter('turn_time_bias_s', 0.00)    # [s]로 미세보정

        # 3) 회전 후 직진 (ㄱ자 꼬리)
        self.declare_parameter('after_turn_rpm',   40)
        self.declare_parameter('after_turn_time_s',1.0)

        # 4) 정지 유지 후 종료
        self.declare_parameter('post_stop_ms',     200)

        # ----- 로드 -----
        self.r   = float(self.get_parameter('wheel_radius_m').value)
        self.W   = float(self.get_parameter('track_width_m').value)
        self.max_rpm   = int(self.get_parameter('max_wheel_rpm').value)
        self.hz        = float(self.get_parameter('output_hz').value)
        self.left_topic  = str(self.get_parameter('left_topic').value)
        self.right_topic = str(self.get_parameter('right_topic').value)
        self.rpm_scale = float(self.get_parameter('rpm_scale').value)

        self.left_polarity  = int(self.get_parameter('left_polarity').value)
        self.right_polarity = int(self.get_parameter('right_polarity').value)

        self.straight_rpm   = int(self.get_parameter('straight_rpm').value)
        self.straight_time  = float(self.get_parameter('straight_time_s').value)

        self.turn_side      = str(self.get_parameter('turn_side').value).lower()
        self.turn_rpm       = int(self.get_parameter('turn_rpm').value)
        self.turn_angle_deg = float(self.get_parameter('turn_angle_deg').value)
        self.turn_gain      = float(self.get_parameter('turn_time_gain').value)
        self.turn_bias_s    = float(self.get_parameter('turn_time_bias_s').value)

        self.after_rpm      = int(self.get_parameter('after_turn_rpm').value)
        self.after_time     = float(self.get_parameter('after_turn_time_s').value)

        self.post_stop_ms   = int(self.get_parameter('post_stop_ms').value)

        # 퍼블리셔
        self.pub_l = self.create_publisher(Int32, self.left_topic,  10)
        self.pub_r = self.create_publisher(Int32, self.right_topic, 10)

        # --- 회전 시간 계산: t = θ * W / (2 * ω * r) ---
        theta = math.radians(self.turn_angle_deg)
        omega = (abs(self.turn_rpm) * 2.0 * math.pi) / 60.0  # 휠 각속도[rad/s]
        base_turn_t = (theta * self.W / (2.0 * omega * self.r)) if omega > 1e-6 else 0.0
        self.turn_time = clamp(base_turn_t * self.turn_gain + self.turn_bias_s, 0.2, 10.0)

        # 상태 머신
        self.state = 'STRAIGHT1'  # → TURN → STRAIGHT2 → STOP
        self.t0 = time.monotonic()

        self._log_params()

        self.timer = self.create_timer(1.0/max(self.hz, 1.0), self.loop)

    def _log_params(self):
        self.get_logger().info(
            f"[RUN] straight1={self.straight_rpm}rpm for {self.straight_time:.2f}s | "
            f"turn({self.turn_side})=±{self.turn_rpm}rpm for {self.turn_time:.2f}s "
            f"(angle={self.turn_angle_deg}°, gain={self.turn_gain}, bias={self.turn_bias_s}s) | "
            f"straight2={self.after_rpm}rpm for {self.after_time:.2f}s"
        )
        self.get_logger().info(
            f"[PARAM] r={self.r}m, W={self.W}m, max_rpm={self.max_rpm}, hz={self.hz}, "
            f"topics=({self.left_topic}, {self.right_topic}), scale={self.rpm_scale}, "
            f"polarity(L,R)=({self.left_polarity},{self.right_polarity})"
        )

    def loop(self):
        now = time.monotonic()

        if self.state == 'STRAIGHT1':
            out_l = self.left_polarity  * self.straight_rpm
            out_r = self.right_polarity * self.straight_rpm
            if (now - self.t0) >= self.straight_time:
                self.state = 'TURN'; self.t0 = now

        elif self.state == 'TURN':
            if self.turn_side == 'left':
                # 좌회전: 좌(-), 우(+)
                out_l = self.left_polarity  * (-self.turn_rpm)
                out_r = self.right_polarity * (+self.turn_rpm)
            else:
                # 우회전: 좌(+), 우(-)
                out_l = self.left_polarity  * (+self.turn_rpm)
                out_r = self.right_polarity * (-self.turn_rpm)
            if (now - self.t0) >= self.turn_time:
                self.state = 'STRAIGHT2'; self.t0 = now

        elif self.state == 'STRAIGHT2':
            out_l = self.left_polarity  * self.after_rpm
            out_r = self.right_polarity * self.after_rpm
            if (now - self.t0) >= self.after_time:
                self.state = 'STOP'; self.t0 = now
        else:  # STOP
            out_l = 0; out_r = 0
            if (now - self.t0) * 1000.0 >= self.post_stop_ms:
                self.get_logger().info("[DONE] L-route complete")
                rclpy.shutdown()
                return

        # 스케일 & 클램프(실내 안전)
        out_l = int(clamp(int(out_l * self.rpm_scale), -self.max_rpm, self.max_rpm))
        out_r = int(clamp(int(out_r * self.rpm_scale), -self.max_rpm, self.max_rpm))

        self.pub_l.publish(Int32(data=out_l))
        self.pub_r.publish(Int32(data=out_r))

def main():
    rclpy.init()
    node = ScriptRunner()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
