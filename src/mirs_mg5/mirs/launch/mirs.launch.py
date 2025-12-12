import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import LaunchConfigurationEquals, IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # パッケージのインストールディレクトリを取得 (推奨される方法)
    # これを使うには CMakeLists.txt で config フォルダが install されている必要があります
    pkg_share = get_package_share_directory('mirs')

    # --- 引数の定義 ---
    esp_port = DeclareLaunchArgument(
        'esp_port', default_value='/dev/ttyUSB1',
        description='Set esp32 usb port.')
    
    lidar_port = DeclareLaunchArgument(
        'lidar_port', default_value='/dev/ttyUSB0',
        description='Set lidar usb port.')

    lidar_model = DeclareLaunchArgument(
        'lidar_model', default_value='s1',
        description='Model of the LiDAR (s1, a1, a2, a3)')

    enable_micro_ros = DeclareLaunchArgument(
        'enable_micro_ros', default_value='true',
        description='Enable micro_ros_agent.')
    
    # --- 設定ファイルのパス ---
    # 既存の設定ファイル
    config_file_path = os.path.join(pkg_share, 'config', 'config.yaml')
    
    # ★追加: EKF用の設定ファイル (ekf.yaml)
    ekf_config_path = os.path.join(pkg_share, 'config', 'ekf_params.yaml')

    # --- ノードの定義 ---

    # 1. オドメトリ配信ノード (注意: C++側でTF配信をOFFにすること！)
    odometry_node = Node(
        package='mirs',
        executable='odometry_publisher',
        name='odometry_publisher',
        output='screen',
        parameters=[config_file_path]
    )

    # 2. パラメータ管理ノード
    parameter_node = Node(
        package='mirs',
        executable='parameter_publisher',
        name='parameter_publisher',
        output='screen',
        parameters=[config_file_path]
    )

    # 3. Micro-ROS Agent
    micro_ros = Node(
        package='micro_ros_agent',
        executable='micro_ros_agent',
        name='micro_ros_agent',
        output='screen',
        arguments=['serial', '--dev', LaunchConfiguration('esp_port'), '-v6'],
        condition=IfCondition(LaunchConfiguration('enable_micro_ros'))
    )

    # 4. LiDARドライバ
    # S1 Model
    sllidar_launch_s1 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('sllidar_ros2'), 'launch', 'sllidar_s1_launch.py')
        ),
        launch_arguments={'serial_port': LaunchConfiguration('lidar_port')}.items(),
        condition=LaunchConfigurationEquals('lidar_model', 's1')
    )

    # A1 Model
    sllidar_launch_a1 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('sllidar_ros2'), 'launch', 'sllidar_a1_launch.py')
        ),
        launch_arguments={'serial_port': LaunchConfiguration('lidar_port')}.items(),
        condition=LaunchConfigurationEquals('lidar_model', 'a1')
    )
    
    # A2 Model (A2M8 is common, using sllidar_a2m8_launch.py)
    sllidar_launch_a2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('sllidar_ros2'), 'launch', 'sllidar_a2m8_launch.py')
        ),
        launch_arguments={'serial_port': LaunchConfiguration('lidar_port')}.items(),
        condition=LaunchConfigurationEquals('lidar_model', 'a2')
    )
    
    # A3 Model
    sllidar_launch_a3 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('sllidar_ros2'), 'launch', 'sllidar_a3_launch.py')
        ),
        launch_arguments={'serial_port': LaunchConfiguration('lidar_port')}.items(),
        condition=LaunchConfigurationEquals('lidar_model', 'a3')
    )

    # 5. Static TF (Base Link -> Laser)
    tf2_ros_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        output='screen',
        arguments=["0", "0", "0.3", "0", "0", "0", "base_link", "laser"]
    )

    # ★追加: robot_localization (EKF) ノード
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[ekf_config_path],
        # 必要な場合はここでトピックのリマップを行います
        # remappings=[('/odometry/filtered', '/odom_combined')] 
    )

    # --- 起動リストの作成 ---
    ld = LaunchDescription()
    
    ld.add_action(esp_port)
    ld.add_action(lidar_port)
    ld.add_action(lidar_model)
    ld.add_action(enable_micro_ros)

    ld.add_action(odometry_node)
    ld.add_action(parameter_node)
    ld.add_action(micro_ros)
    
    ld.add_action(sllidar_launch_s1)
    ld.add_action(sllidar_launch_a1)
    ld.add_action(sllidar_launch_a2)
    ld.add_action(sllidar_launch_a3)
    
    ld.add_action(tf2_ros_node)
    
    # ★追加: EKFを起動リストに追加
    # Debug: Fake Odom when micro_ros is disabled
    fake_odom_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        output='screen',
        arguments=["0", "0", "0", "0", "0", "0", "odom", "base_link"],
        condition=UnlessCondition(LaunchConfiguration('enable_micro_ros'))
    )

    ld.add_action(ekf_node)
    
    ld.add_action(fake_odom_node)

    return ld
