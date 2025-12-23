#!/usr/bin/env python3
import time
import sys
import signal
import argparse

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32


class SequenceRpmRunner(Node):
    def __init__(self, steps, hz=20.0, left_topic='/cmd_rpm_left', right_topic='/cmd_rpm_right'):
        super().__init__('sequence_rpm_runner')
        self.pub_l = self.create_publisher(Int32, left_topic, 10)
        self.pub_r = self.create_publisher(Int32, right_topic, 10)
        self.steps = steps
        self.hz = hz
        signal.signal(signal.SIGINT, self._sigint)

    def _pub_pair(self, l, r):
        ml = Int32(); ml.data = l
        mr = Int32(); mr.data = r
        self.pub_l.publish(ml)
        self.pub_r.publish(mr)

    def _stop(self):
        stop = Int32(); stop.data = 0
        for _ in range(3):  # 드롭 방지로 몇 번 더 보냄
            self.pub_l.publish(stop)
            self.pub_r.publish(stop)
            time.sleep(0.05)

    def _sigint(self, *_):
        self.get_logger().info('CTRL-C -> stop(0)')
        self._stop()
        rclpy.shutdown()
        sys.exit(0)

    def run(self):
        period = 1.0 / self.hz
        for dur, l, r in self.steps:
            self.get_logger().info(f'step: {dur:.2f}s | L={l} rpm, R={r} rpm')
            t_end = time.time() + dur
            while time.time() < t_end:
                self._pub_pair(l, r)
                rclpy.spin_once(self, timeout_sec=0.0)
                time.sleep(period)
        self.get_logger().info('sequence done -> stop(0)')
        self._stop()


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--hz', type=float, default=20.0, help='publish rate (Hz)')
    parser.add_argument('--left-topic', default='/cmd_rpm_left')
    parser.add_argument('--right-topic', default='/cmd_rpm_right')
    # 기본 시퀀스: (duration, L_rpm, R_rpm)
    parser.add_argument('--t1', type=float, default=3.0); parser.add_argument('--l1', type=int, default=-10); parser.add_argument('--r1', type=int, default=-10)
    # parser.add_argument('--t2', type=float, default=2.0); parser.add_argument('--l2', type=int, default=-5);  parser.add_argument('--r2', type=int, default=-10)
    # parser.add_argument('--t3', type=float, default=2.0); parser.add_argument('--l3', type=int, default=-10); parser.add_argument('--r3', type=int, default=-5)
    cli_args, ros_args = parser.parse_known_args()

    steps = [
        (cli_args.t1, cli_args.l1, cli_args.r1),
        # (cli_args.t2, cli_args.l2, cli_args.r2),
        # (cli_args.t3, cli_args.l3, cli_args.r3),
    ]

    rclpy.init(args=ros_args)
    node = SequenceRpmRunner(steps, hz=cli_args.hz,
                             left_topic=cli_args.left_topic,
                             right_topic=cli_args.right_topic)
    try:
        node.run()
    finally:
        node._stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
