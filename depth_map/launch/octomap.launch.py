from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 1) k4a 드라이버는 point_cloud=true 로 띄운 상태라고 가정
        Node(
            package='octomap_server',
            executable='octomap_server_node',
            name='octomap_server',
            output='screen',
            parameters=[
                { 'frame_id': 'camera_base' },
                { 'sensor_model/max_range': 10.0 },
                { 'resolution': 0.1 },
            ],
            remappings=[
                ('/cloud_in', '/k4a/points2'),
            ]
        ),
    ])
