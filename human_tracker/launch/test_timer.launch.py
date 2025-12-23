#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import LogInfo, TimerAction

def generate_launch_description():
    return LaunchDescription([
        LogInfo(msg='[Test] Launch started!'),
        TimerAction(
            period=2.0,  # 2초 뒤 실행
            actions=[
                LogInfo(msg='[Test] Timer expired—this should appear after 2 seconds'),
            ],
        ),
    ])
