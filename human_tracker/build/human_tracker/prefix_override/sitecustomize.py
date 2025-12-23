import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/go/ros2_ws/src/human_tracker/install/human_tracker'
