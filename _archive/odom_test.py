import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time
import math

# --- 設定項目 ---
TARGET_ANGLE_DEG = 360.0 # (目標角度: 360 度)
ANGULAR_SPEED_RADPS = 0.2 # (回転速度: 0.1 rad/s)
# --- ---

# 角度と角速度から実行時間を計算
TARGET_ANGLE_RAD = math.radians(TARGET_ANGLE_DEG)
DURATION_SEC = TARGET_ANGLE_RAD / ANGULAR_SPEED_RADPS

class MoveTestNode(Node):
    def __init__(self):
        super().__init__('move_test_node')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.05, self.publish_velocity) # 20Hzでパブリッシュ
        self.start_time = self.get_clock().now()
        self.end_time = self.start_time + rclpy.duration.Duration(seconds=DURATION_SEC)
        
        self.get_logger().info(f"旋回テスト開始: {ANGULAR_SPEED_RADPS:.2f} rad/s で {DURATION_SEC:.1f} 秒間（{TARGET_ANGLE_DEG:.1f}度）旋回します。")
        self.test_finished = False

    def publish_velocity(self):
        current_time = self.get_clock().now()
        
        msg = Twist()
        
        if current_time < self.end_time:
            # まだ時間内: 旋回コマンドを送信
            msg.linear.x = 0.0
            msg.angular.z = ANGULAR_SPEED_RADPS
        else:
            # 時間超過: 停止コマンドを送信し続ける
            msg.linear.x = 0.0
            msg.angular.z = 0.0
            
            # 終了メッセージを一度だけ表示
            if not self.test_finished:
                self.get_logger().info(f"テスト時間 {DURATION_SEC:.1f} 秒が経過。停止コマンドを送信中...")
                self.get_logger().info("Ctrl+C を押してスクリプトを終了してください。")
                self.test_finished = True
        
        # ウォッチドッグが作動しないよう、常にメッセージを送信する
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = MoveTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            # 終了時に必ず停止コマンドを送信
            stop_msg = Twist()
            stop_msg.linear.x = 0.0
            stop_msg.angular.z = 0.0
            node.publisher.publish(stop_msg)
            time.sleep(0.1) 
            node.get_logger().info("スクリプトをシャットダウンします。")
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()