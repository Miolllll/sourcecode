from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'human_tracker'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),  # 자동으로 human_tracker/ 내부 모듈 탐지
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # Nav2 parameters 파일 설치
        (os.path.join('share', package_name, 'config'), ['config/nav2_params.yaml', 'config/run_all.rviz',]),
    ],
    install_requires=[
        'setuptools',
        'torch',
        'opencv-python',
        'ultralytics',
        'numpy',
        'torchvision',
    ],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your@email.com',
    description='Human tracking node using YOLOv11 pose + DeepSORT + Azure Kinect',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'human_tracker_node = human_tracker.human_tracker_node:main',
            'user_goal_planner = human_tracker.user_goal_planner_node:main',
            'path_follower_node = human_tracker.path_follower_node:main',
            'user_goal_planner_safe = human_tracker.user_goal_planner_safe:main',
        ],
    },
)
