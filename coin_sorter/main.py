import cv2
import json
import torch
import serial
import time
import numpy as np
from collections import deque
from PIL import Image
from torchvision import transforms
from model import SimpleCNN, IMG_SIZE

# 設定ファイルのパス
CONFIG_PATH = "config.json"

class FaderSystem:
    def __init__(self, name, config):
        self.name = name
        self.cam_id = config["cam_id"]
        self.com_port = config["com_port"]
        self.model_path = config["model_path"]
        self.roi_path = config["roi_path"]
        
        # ROI設定 (監視エリア) の読み込み
        try:
            with open(self.roi_path, 'r') as f:
                self.roi = json.load(f)
            print(f"[{name}] ROI Loaded: {self.roi}")
        except FileNotFoundError:
            print(f"[{name}] Error: ROI Check failed ({self.roi_path}). Run calibration first.")
            self.roi = None

        # AIモデルの読み込み
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleCNN().to(self.device)
        try:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            print(f"[{name}] Model Loaded: {self.model_path}")
        except FileNotFoundError:
            print(f"[{name}] Error: Model not found ({self.model_path}). Run train.py.")
            self.model = None

        # 画像変換 (AI用)
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ])

        # シリアル通信 (Arduino接続)
        self.ser = None
        try:
            self.ser = serial.Serial(self.com_port, 9600, timeout=1)
            time.sleep(2) # Arduinoのリセット待機
            print(f"[{name}] Serial Connected: {self.com_port}")
        except Exception as e:
            print(f"[{name}] Serial Connection Failed: {e}")

        # カメラ設定
        # WindowsではDirectShow (CAP_DSHOW) が高速で安定することが多い
        print(f"[{name}] Opening Camera {self.cam_id} (DSHOW)...")
        self.cap = cv2.VideoCapture(self.cam_id, cv2.CAP_DSHOW)
        
        if not self.cap.isOpened():
             print(f"[{name}] DSHOW Failed. Trying default backend...")
             self.cap = cv2.VideoCapture(self.cam_id)
             
             if not self.cap.isOpened():
                 print(f"[{name}] Camera {self.cam_id} failed. Trying ID 0...")
                 self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if self.cap.isOpened():
             print(f"[{name}] Camera Opened Successfully.")

        # 状態管理 (チャタリング防止用)
        self.history = deque(maxlen=3) # 過去3フレームの結果を保持
        self.last_sent_cmd = None

    def process(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, "Cam Error"

        # ROIが設定済みならAI判定を実行
        status_text = "Init"
        color = (0, 255, 255) # 黄色

        if self.roi and self.model:
            x, y, w, h = self.roi["x"], self.roi["y"], self.roi["w"], self.roi["h"]
            
            # ROIが画面外にはみ出さないように補正
            ih, iw, _ = frame.shape
            x = max(0, min(x, iw-1))
            y = max(0, min(y, ih-1))
            w = max(1, min(w, iw-x))
            h = max(1, min(h, ih-y))

            # 切り抜きと推論
            crop = frame[y:y+h, x:x+w]
            # BGRからRGBへ変換
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            input_tensor = self.transform(crop_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                _, predicted = torch.max(outputs, 1)
                label = predicted.item() 
                
                # ラベル 0: 'o' (コインなし) -> 動かす -> '1'を送る
                # ラベル 1: '1' (コインあり) -> 止める -> '0'を送る
                self.history.append(label)

            # チャタリング防止 (3回連続で同じ結果なら確定)
            if len(self.history) == 3:
                # 全て 1 (コインあり) -> ストップ
                if all(x == 1 for x in self.history): 
                    target_cmd = '0' # Stop
                    status_text = "COIN (Stop)"
                    color = (0, 0, 255) # 赤
                # 全て 0 (コインなし) -> スタート
                elif all(x == 0 for x in self.history): 
                    target_cmd = '1' # Start
                    status_text = "EMPTY (Active)"
                    color = (0, 255, 0) # 緑
                else:
                    target_cmd = self.last_sent_cmd # 状態維持
                    status_text = "Unstable"
                
                # コマンド送信 (状態が変わった時のみ)
                if target_cmd != self.last_sent_cmd and self.ser:
                    try:
                        # 改行コードを付与して送信
                        self.ser.write((target_cmd + '\n').encode())
                        self.last_sent_cmd = target_cmd
                        print(f"[{self.name}] Sent: {target_cmd}")
                    except Exception as e:
                        print(f"[{self.name}] Write Error: {e}")

            # 画面描画
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, status_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        return frame, status_text

    def calibrate_roi(self):
        """インタラクティブなROI設定ツール"""
        print(f"[{self.name}] Starting Calibration... Press SPACE/ENTER to confirm.")
        if not self.cap.isOpened():
             print(f"[{self.name}] Camera not open.")
             return

        # 1フレーム読み込み
        ret, frame = self.cap.read()
        if not ret:
             print(f"[{self.name}] Failed to grab frame.")
             return

        # ROI選択ウィンドウ表示
        # showCrosshair=True, fromCenter=False
        r = cv2.selectROI("ROI Selection", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("ROI Selection")
        
        # r は (x, y, w, h)
        if r[2] > 0 and r[3] > 0: # 幅と高さが有効なら保存
            self.roi = {"x": int(r[0]), "y": int(r[1]), "w": int(r[2]), "h": int(r[3])}
            print(f"[{self.name}] New ROI: {self.roi}")
            
            # ファイルへ保存
            with open(self.roi_path, 'w') as f:
                json.dump(self.roi, f)
            print(f"[{self.name}] ROI Saved to {self.roi_path}")
        else:
            print(f"[{self.name}] Selection Cancelled.")

    def release(self):
        if self.cap.isOpened(): self.cap.release()
        if self.ser: self.ser.close()

def main():
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    # システム初期化
    sys_a = FaderSystem("System A", config["system_a"])

    print("--- Coin Sorter AI System Started ---")
    print("Keys:")
    print(" 'c': Calibrate ROI")
    print(" 'q': Quit")

    # 起動時のキャリブレーション確認
    print("Auto-start in 3s... ('c' to Calibrate, 's' to Skip)")
    
    start_time = time.time()
    while True:
        frame, _ = sys_a.process() # カメラ映像を表示確認用に出す
        if frame is None: break
        
        # 経過時間
        elapsed = time.time() - start_time
        remaining = 3.0 - elapsed
        
        if remaining <= 0:
            print("Auto Starting.")
            break

        # 画面上に操作ガイドを表示
        msg = f"Auto Start in {remaining:.1f}s... Press 'c' to Calibrate"
        cv2.putText(frame, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Coin Sorter Monitor", frame)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('c'):
             sys_a.calibrate_roi()
             break 
        elif key == ord('s') or key == 13: # 's' or Enter
             print("Manual Start.")
             break
        elif key == ord('q'):
             sys_a.release()
             cv2.destroyAllWindows()
             return

    # メインループ
    try:
        while True:
            frame_a, status_a = sys_a.process()

            if frame_a is not None:
                cv2.imshow("Coin Sorter Monitor", frame_a)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'): # 実行中も調整可能
                sys_a.calibrate_roi()
    
    except KeyboardInterrupt:
        pass
    finally:
        sys_a.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
