#!/usr/bin/env python3

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import TimerAction, LogInfo

def generate_launch_description():
    # 패키지 경로
    pkg_share   = get_package_share_directory('human_tracker')
    nav2_share  = get_package_share_directory('nav2_bringup')
    slam_share  = get_package_share_directory('slam_toolbox')
    nav2_params = os.path.join(pkg_share, 'config', 'nav2_params.yaml')

    rviz_config = os.path.join(pkg_share, 'config', 'run_all.rviz')

    # # RViz 설정 파일 (Nav2 기본 뷰)
    # rviz_config = os.path.join(nav2_share, 'rviz', 'nav2_default_view.rviz')

    # 0) Nav2 파라미터 파일 인자 선언
    declare_nav2_arg = DeclareLaunchArgument(
        'params_file',
        default_value=nav2_params,
        description='Nav2 파라미터 파일 경로'
    )

    # 1) SLAM Toolbox (online_async) → /map 토픽 퍼블리시
    slam_toolbox = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam_share, 'launch', 'online_async_launch.py')
        ),
        launch_arguments=[
            ('use_sim_time', 'False'),
        ]
    )

    # 3) Azure Kinect 드라이버
    azure_kinect = Node(
        package='azure_kinect_ros2_driver',
        executable='azure_kinect_node',
        name='k4a_ros2_node',
        output='screen',
        emulate_tty=True,
        parameters=[
            {'color_enabled': True},
            {'color_resolution': '720P'},
            {'color_format': 'bgra'},
            {'fps': 30},
            {'depth_enabled': True},
            {'depth_mode': 'NFOV_UNBINNED'},
            {'point_cloud': True},
            {'rgb_point_cloud': False},
            {'point_cloud_in_depth_frame': True},
        ],
    )

    # 4) PointCloud 필터링 노드
    filter_node = Node(
        package='depth_map',
        executable='filter_node',
        name='point_cloud_filter_node',
        output='screen',
        parameters=[
            {'input_cloud_topic': '/k4a/points2'},
            {'filtered_cloud_topic': '/filtered_points'},
        ],
    )

    # 5) 2D 맵핑 노드
    mapping_node = Node(
        package='depth_map',
        executable='pointcloud_2d_map_node',
        name='pointcloud_2d_map_node',
        output='screen',
        parameters=[
            {'input_cloud_topic': '/filtered_points'},
            {'map_topic': '/pointcloud_2d_map'},
            {'frame_id': 'camera_base'},
            {'resolution': 0.08},
            {'width': 200},
            {'height': 200},
            {'z_min': 0.2},
            {'z_max': 3.5},
        ],
    )

    # 6) Human Tracker 노드
    human_tracker = Node(
        package='human_tracker',
        executable='human_tracker_node',
        name='human_tracker',
        output='screen',
        parameters=[
            {'rgb_topic': '/k4a/rgb/image_raw'},
            {'depth_topic': '/k4a/depth_to_rgb/image_raw'},
            {'camerainfo_topic': '/k4a/depth/camera_info'},
            {'user_topic': '/user_tracking'},
            {'frame_id': 'camera_base'},
            {'fx': 525.0},
            {'fy': 525.0},
            {'cx': 320.0},
            {'cy': 240.0},
        ],
    )

    # # 7) User Goal Planner 노드
    # user_goal_planner = Node(
    #     package='human_tracker',
    #     executable='user_goal_planner',
    #     name='user_goal_planner',
    #     output='screen',
    #     parameters=[
    #         {'map_topic': '/pointcloud_2d_map'},
    #         {'goal_topic': '/user_tracking'},
    #         {'path_topic': '/global_plan'},
    #         {'base_frame': 'camera_base'},
    #         {'start_x': 0.0},
    #         {'start_y': 0.0},
    #     ],
    # )

    user_goal_planner_safe = Node(
    package='human_tracker',
    executable='user_goal_planner_safe',
    name='user_goal_planner_safe',
    output='screen',
    parameters=[{
            # --- 센서/맵 신뢰성 & 노이즈 억제 ---
            'hit_threshold': 9,            # 3~5
            'age_threshold_frames': 0,      # 3~5
            'occ_threshold': 65,            # 60~70
            'unknown_is_lethal': True,      # 필요시 False로 테스트

            # --- 사용자 마스킹 ---
            'user_mask_radius_m': 0.5,

            # --- 로봇 크기 & 오프셋(후륜축 시작점) ---
            'robot_width_m': 0.90,
            'robot_length_m': 1.20,
            'use_footprint_offset': True,
            'footprint_offset_x_m': -0.87,
            'footprint_offset_y_m':  0.0,

            # --- 팽창: 통과성 우선 (width) ---
            'inflation_mode': 'width',
            'side_margin_m': 0.06,          # 0.03~0.05
            'gap_relax_cells': 1,           # 2~3

            # --- 선택적 팽창: 작은 점은 팽창 금지 ---
            'use_selective_inflation': True,
            'min_inflate_blob_area_m2': 0.02,   # 0.008~0.02 (3~5셀부터만 크게 팽창)

            # --- 회전-클리어런스 비용 (NEW) ---
            'use_turn_clearance_cost': True,
            'turn_clearance_margin_m': 0.12,     # 모서리 여유 5cm
            'turn_clearance_gain': 1.2,          # 1.5~3.0
            'clearance_power': 2.0,              # 제곱 패널티

            # --- 경로 품질(지그재그 억제 & 자연스러운 곡선) ---
            'turn_penalty': 0.3,
            'rdp_epsilon_m': 0.25,
            'smooth_iterations': 2,

            # --- 가운데 치우침 완화 (벽을 더 타게) ---
            'use_center_bias': True,
            'center_bias_weight': 0.05,
            'center_bias_gamma': 1.0,

            # --- 주기 ---
            'tick_period_s': 0.30,          # 0.25~0.35
    }],
)

    # 8) Path Follower 노드 (지연 없이 바로 실행)
    path_follower = Node(
        package='human_tracker',
        executable='path_follower_node',   # setup.py의 console_scripts 등록명과 일치
        name='path_follower',
        output='screen',
        parameters=[{
            'path_topic': '/global_plan',
            'user_topic': '/user_tracking',
            'cmd_topic':  '/cmd_vel',

            # 'lookahead_distance': 0.8,
            'target_distance':    1.4,

            'min_linear_speed':   0.15,
            'max_linear_speed':   1.65,
            'max_angular_speed':  2.50,

            'assume_robot_at_origin': True,
            'assume_heading_plus_x':  True,

            # --- frames ---
            'base_frame': 'rear_axle',          # 추종/조향 기준
            'distance_frame': 'camera_base',    # 사람-거리 기준 (원하면)

            # --- auto static TF (camera_base -> rear_axle) ---
            'publish_rear_axle_tf': True,
            'camera_frame': 'camera_base',
            'rear_axle_offset_x': -0.87,        # RViz에서 확인한 그대로
            'rear_axle_offset_y':  0.0,
            'rear_axle_offset_z':  0.0,
            'rear_axle_yaw_rpy':   0.0,
        }],
    )

    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config],  
        parameters=[{'use_sim_time': False}],
    )

    md_controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('md_controller'),
                'launch',
                'md_controller.launch.py'
            )
        ),
        launch_arguments=[
            ('use_sim_time', 'False'),
            # 필요 시 md_controller의 파라미터/인자 추가:
            # ('port', '/dev/ttyUSB0'),
            # ('baudrate', '115200'),
            # ('invert_left', 'true'),
            # ('invert_right','false'),
            # ('left_topic', '/cmd_rpm_left'),
            # ('right_topic','/cmd_rpm_right'),
        ]
    )

    # C) 모터 하드웨어 노드 (motor_controller)
    motor_node = Node(
        package='motor_controller',
        executable='motor_node',     # motor_ws/motor_controller의 실행 파일명
        name='motor_node',
        output='screen',
        emulate_tty=True,
        parameters=[{
            # 패키지에 맞춰 필요한 파라미터가 있으면 여기에 채우세요.
            # 예: 'left_topic': '/cmd_rpm_left', 'right_topic': '/cmd_rpm_right'
        }],
        # remappings=[  # 필요 시 주석 해제
        #     ('/cmd_rpm_left',  '/cmd_rpm_left'),
        #     ('/cmd_rpm_right', '/cmd_rpm_right'),
        # ]
    )

    return LaunchDescription([
        declare_nav2_arg,
        slam_toolbox,
        azure_kinect,
        filter_node,
        mapping_node,
        human_tracker,
        # user_goal_planner,
        user_goal_planner_safe,
        path_follower,  
        rviz2,
        md_controller_launch,
        motor_node,
    ])
