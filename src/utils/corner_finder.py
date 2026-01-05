import cv2
import numpy as np

# ===============================================================
# 碁盤の四隅の座標を取得するための補助プログラム
# ===============================================================

# --- 設定 ---
CAM_INDEX = 2 # メインプログラムと同じカメラ番号に設定してください

# --- グローバル変数 ---
corner_points = []
window_name = "Corner Finder - Click corners (TL -> TR -> BR -> BL), then press 'p'"

def mouse_callback(event, x, y, flags, param):
    """マウスクリックを処理し、角の座標を保存する"""
    global corner_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(corner_points) < 4:
            corner_points.append((x, y))
            print(f"  > {len(corner_points)}番目の角を ({x}, {y}) に設定。")
        else:
            print("⚠️ 4つの角は設定済みです。「p」キーで座標を出力してください。")

def main():
    global corner_points
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"❌ カメラ({CAM_INDEX})の起動に失敗しました。")
        return

    print("--- 碁盤の四隅 座標取得ツール ---")
    print("1. ウィンドウ上で碁盤の 左上 -> 右上 -> 右下 -> 左下 の順にクリックしてください。")
    print("2. 4点クリック後、「p」キーを押して座標をコンソールに出力します。")
    print("3. 出力された行をコピーして、メインプログラムに貼り付けてください。")
    print("   - 'r'キー: やり直し")
    print("   - 'q'キー: 終了")

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        display_frame = frame.copy()

        # クリックした点を描画
        for i, point in enumerate(corner_points):
            cv2.circle(display_frame, point, 7, (0, 0, 255), -1)
            cv2.putText(display_frame, str(i+1), (point[0]+10, point[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("👋 終了します。")
            break
        
        if key == ord('r'):
            corner_points = []
            print("🔄 リセットしました。もう一度クリックしてください。")
        
        if key == ord('p'):
            if len(corner_points) == 4:
                print("\n✅ 座標が確定しました！")
                print("↓ この行をコピーして、メインプログラムの指定箇所に貼り付けてください ↓")
                print(f"FIXED_CORNER_POINTS = {corner_points}")
                print("-" * 60)
            else:
                print(f"⚠️ 角が4つ指定されていません。(現在: {len(corner_points)}個)")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()