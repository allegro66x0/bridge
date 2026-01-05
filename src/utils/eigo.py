import keyboard
import pyautogui
import time
import random
import sys

# ---------------------------------------------------------
# エッセイの内容（A判定狙い版）
# ---------------------------------------------------------
my_essay = """ここに文章を入力"""
# ---------------------------------------------------------

# コナミコマンドの定義
KONAMI_CODE = ['up', 'up', 'down', 'down', 'left', 'right', 'left', 'right', 'b', 'a']
current_history = []

# 一時停止に使うキー
PAUSE_KEY = 'ctrl'

def auto_type_essay():
    """エッセイを自動入力する関数"""
    print("\n\n★ コナミコマンド検出！自動入力を開始します ★")
    print(f"★ {PAUSE_KEY}キー を押している間は一時停止します ★")
    
    # 直前に入力された 'b' と 'a' を消すためにBackSpaceを2回押す
    pyautogui.press('backspace', presses=2)
    time.sleep(0.5)

    # 入力開始
    for char in my_essay:
        # --- 一時停止機能 ---
        if keyboard.is_pressed(PAUSE_KEY):
            print("\r[一時停止中...]   ", end="")
            while keyboard.is_pressed(PAUSE_KEY):
                time.sleep(0.1) # キーが離されるまで待機
            print("\r[再開]           ", end="")
            # フォーカスが外れている可能性を考慮して少し待つ
            time.sleep(0.5)
        # -------------------

        pyautogui.write(char)
        
        # 緊急停止（マウスを左上にやった場合）
        if pyautogui.position() == (0, 0):
             print("\n緊急停止しました")
             return

        # タイピング速度のゆらぎ
        time.sleep(random.uniform(0.005, 0.03)) 

    print("\n\n--- 入力完了 ---")
    current_history.clear()

def on_key_event(e):
    """キー入力を監視する関数"""
    global current_history

    if e.event_type == 'down':
        key = e.name
        
        # 履歴に追加
        current_history.append(key)
        if len(current_history) > 20:
            current_history.pop(0)

        # コナミコマンド判定
        if current_history[-len(KONAMI_CODE):] == KONAMI_CODE:
            auto_type_essay()

if __name__ == "__main__":
    # 安全装置: マウスを画面の左上にやると止まる設定
    pyautogui.FAILSAFE = True

    print("-------------------------------------------------------")
    print("  監視待機中...")
    print("  ブラウザの入力欄を選択し、以下のコマンドを入力してください。")
    print("  [↑] [↑] [↓] [↓] [←] [→] [←] [→] [b] [a]")
    print(f"  ★ 入力中に [{PAUSE_KEY}] を押し続けると一時停止します")
    print("-------------------------------------------------------")
    print("終了するには Ctrl+C を押してください")

    # キーボード監視を開始
    keyboard.hook(on_key_event)
    
    # 終了しないように待機
    keyboard.wait()