#!/usr/bin/env python3
import rclpy
import sys, signal, os
from rclpy.node import Node
from std_msgs.msg import Int32
from getkey import getkey

MAX_VEL = 1000
VEL_STEP_SIZE = 10

msg = """
-------------------------------------------------
[  : CCW   ]  : CW
0  : STOP

CTRL-C to quit
-------------------------------------------------
"""

class TeleopKey(Node):
    def __init__(self):
        super().__init__("teleop_node")

        qos = rclpy.qos.QoSProfile(depth=10, reliability=rclpy.qos.ReliabilityPolicy.RELIABLE)
        self.pub_left  = self.create_publisher(Int32, "/cmd_rpm_left",  qos)
        self.pub_right = self.create_publisher(Int32, "/cmd_rpm_right", qos)

        signal.signal(signal.SIGINT, self.signal_handler)

        self.target_rpm = 0

        os.system('clear'); print(msg); print("target_RPM =", self.target_rpm)

        while True:
            key = getkey()
            if key == "[":
                self.target_rpm = self._clamp(self.target_rpm - VEL_STEP_SIZE)
            elif key == "]":
                self.target_rpm = self._clamp(self.target_rpm + VEL_STEP_SIZE)
            elif key == "0":
                self.target_rpm = 0
            else:
                continue

            os.system('clear'); print(msg)
            print("target_RPM =", self.target_rpm)
            self.publish_lr(self.target_rpm)

    def publish_lr(self, rpm: int):
        m = Int32(); m.data = rpm
        self.pub_left.publish(m)
        self.pub_right.publish(m)

    def signal_handler(self, sig, frame):
        m = Int32(); m.data = 0
        print("target_RPM = 0 (CTRL-C)")
        self.pub_left.publish(m)
        self.pub_right.publish(m)
        sys.exit(0)

    def _clamp(self, v: int) -> int:
        if v < -MAX_VEL: return -MAX_VEL
        if v >  MAX_VEL: return  MAX_VEL
        return v

def main(args=None):
    rclpy.init(args=args)
    node = TeleopKey()
    rclpy.spin(node)  # while 루프가 blocking이어서 도달하지 않지만 형식상 유지
    rclpy.shutdown()

if __name__ == '__main__':
    main()
