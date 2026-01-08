import serial
import time

# --- 設定 ---
PORT = 'COM10'  # ユーザー環境のGantryポート(COM10)に合わせて変更
BAUDRATE = 9600

def wait_for_ready(ser):
    """ArduinoからのREADY信号待ち"""
    print("待機中...")
    while True:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line == "READY":
                print("-> Arduino Ready!")
                break
            elif line:
                print(f"Arduino: {line}")
        except Exception as e:
            pass

def send_command(ser, x, y):
    """座標送信 (例: x=1, y=13 -> '0113')"""
    # 1-13の範囲であることを確認
    x = max(1, min(13, int(x)))
    y = max(1, min(13, int(y)))
    
    command = f"{x:02}{y:02}\n"
    print(f"送信: ({x}, {y}) -> {command.strip()}")
    ser.write(command.encode('utf-8'))

def main():
    try:
        ser = serial.Serial(PORT, BAUDRATE, timeout=1)
        print(f"接続完了: {PORT}")
        
        # 起動時の原点復帰待ち
        wait_for_ready(ser)
        
        # テストする座標リスト (13路盤)
        targets = [
            (1, 1),    # 左奥
            (13, 1),   # 右奥
            (1, 13),   # 左手前
            (13, 13),  # 右手前
            (7, 7)     # 天元（中央）
        ]
        
        print("テスト開始: 5点動作確認")
        for x, y in targets:
            send_command(ser, x, y)
            wait_for_ready(ser) # 動作完了待ち
            time.sleep(1)
        
        print("テスト完了")

    except serial.SerialException:
        print(f"エラー: {PORT} が開けません。")
    except KeyboardInterrupt:
        print("\n終了します。")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == "__main__":
    main()
