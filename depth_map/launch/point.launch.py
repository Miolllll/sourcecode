from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='depth_map',              # 자신의 패키지 이름으로 바꿔주세요
            executable='point_map_node',         # setup.py 에 등록한 console_scripts 이름
            name='point_map_node',
            output='screen',
            parameters=[
                # Python 노드에서 선언한 파라미터들
                {'frame_id': 'depth_camera_link'},
                {'z_min': 0.05},
                {'z_max': 10.0},
                {'resolution': 0.1},
                {'grid_size': 100},
            ],
            # 필요하다면 remappings:
            # remappings=[
            #     ('/k4a/points2', '/your/pointcloud2_topic'),
            # ],
        ),
    ])
