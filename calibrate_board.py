import cv2
import numpy as np
import re
import os

# 設定
CAM_INDEX = 0  # メインスクリプトと同じにする
TARGET_FILE = "L6/webcam_gomoku_ai.py"

clicked_points = []
img_display = None

def mouse_callback(event, x, y, flags, param):
    global clicked_points, img_display
    if event == cv2.EVENT_LBUTTONDOWN:
        # 表示倍率が2倍なので、実際の座標は1/2にする
        real_x, real_y = x // 2, y // 2
        
        if len(clicked_points) < 4:
            clicked_points.append((real_x, real_y))
            print(f"Point {len(clicked_points)}: ({real_x}, {real_y})")

def update_python_file(points):
    """L6/webcam_gomoku_ai.py の FIXED_CORNER_POINTS を書き換える"""
    if not os.path.exists(TARGET_FILE):
        print(f"エラー: {TARGET_FILE} が見つかりません。")
        return

    try:
        with open(TARGET_FILE, 'r', encoding='utf-8') as f:
            content = f.read()

        # 正規表現で置換 (FIXED_CORNER_POINTS = [ ... ])
        pattern = r"FIXED_CORNER_POINTS\s*=\s*\[.*?\]"
        new_val = f"FIXED_CORNER_POINTS = {points}"
        
        new_content = re.sub(pattern, new_val, content, count=1)

        with open(TARGET_FILE, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"✅ {TARGET_FILE} を更新しました！: {new_val}")

    except Exception as e:
        print(f"❌ ファイル更新エラー: {e}")

def main():
    global img_display, clicked_points
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("カメラが開けません")
        return

    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", mouse_callback)

    print("="*50)
    print("【盤面キャリブレーションツール】(2倍 拡大表示モード)")
    print("盤面の四隅を順番にクリックしてください（左上 → 右上 → 右下 → 左下）")
    print("4点クリックしたら 's' キーで保存して終了します。")
    print("'r' キーでリセット、'q' キーで保存せずに終了します。")
    print("="*50)

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 表示用に2倍に拡大
        h, w = frame.shape[:2]
        img_display = cv2.resize(frame, (w*2, h*2))

        # クリックした点を描画 (表示座標系で描画)
        for i, pt in enumerate(clicked_points):
            disp_pt = (pt[0]*2, pt[1]*2)
            cv2.circle(img_display, disp_pt, 10, (0, 0, 255), -1) # 円も少し大きく
            cv2.putText(img_display, str(i+1), (disp_pt[0]+20, disp_pt[1]-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # 4点揃ったら四角形を描画
        if len(clicked_points) == 4:
            pts = np.array([(p[0]*2, p[1]*2) for p in clicked_points], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img_display, [pts], True, (0, 255, 0), 3)
            
            cv2.putText(img_display, "Press 's' to SAVE", (40, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)

        cv2.imshow("Calibration", img_display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('s'):
            if len(clicked_points) == 4:
                print("保存中...")
                update_python_file(clicked_points)
                break
            else:
                print("まだ4点選択されていません！")
        
        elif key == ord('r'):
            clicked_points = []
            print("リセットしました")

        elif key == ord('q'):
            print("保存せずに終了します")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
