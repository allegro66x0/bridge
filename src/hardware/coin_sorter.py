import serial
import time
import threading

class CoinSorter:
    def __init__(self, port, baud_rate=9600):
        self.port = port
        self.baud_rate = baud_rate
        self.ser = None
        self.is_connected = False
        self.lock = threading.Lock() # シリアル通信の競合を防ぐ

    def connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baud_rate, timeout=1)
            # Arduinoのリセット待ち
            time.sleep(2) 
            self.is_connected = True
            print(f"[CoinSorter] Connected to {self.port}")
            return True
        except serial.SerialException as e:
            print(f"[CoinSorter] Connection Failed: {e}")
            self.is_connected = False
            return False

    def close(self):
        if self.ser and self.ser.is_open:
            self.stop_all() # 安全のため停止
            self.ser.close()
            self.is_connected = False
            print("[CoinSorter] Disconnected")

    def _send_command(self, cmd):
        if not self.is_connected or not self.ser:
            return
        
        with self.lock:
            try:
                full_cmd = f"{cmd}\n"
                self.ser.write(full_cmd.encode('utf-8'))
                # 必要ならレスポンス待機を入れる
            except Exception as e:
                print(f"[CoinSorter] Send Error: {e}")

    def start_all(self):
        """すべてのモータをデフォルト速度で駆動（供給開始）"""
        self._send_command("START")

    def stop_all(self):
        """すべてのモータを停止"""
        self._send_command("STOP")

    def set_feeder_speed(self, speed):
        """フェーダー(M1)の速度設定 (0-255)"""
        self._send_command(f"M1:{int(speed)}")

    def set_conveyor_speed(self, speed):
        """コンベア(M2, M3)の速度設定 (0-255)"""
        self._send_command(f"M2:{int(speed)}")
        self._send_command(f"M3:{int(speed)}")
