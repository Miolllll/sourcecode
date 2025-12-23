from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='depth_map',
            executable='depth_map_node',
            name='depth_map_node',
            output='screen',
            emulate_tty=True,
            parameters=[
                {"frame_id": "depth_camera_link" },
            ],
            remappings=[
                ('/camera/depth/image_rect_raw', '/k4a/depth/image_raw'),
                ('/camera/depth/camera_info',     '/k4a/depth/camera_info'),
            ]
        )
    ])
