import subprocess
import sys
import os

# --- ユーザー設定エリア ---
# "番号": ("メニューに表示する名前", "実行したいファイル名.py")
models = {
    # ↓↓ (表示名, ファイル名) の形で編集してください ↓↓
    "1": ("AI vs AI", "tests/AIvsAI/ai_battle.py"),
    "2": ("連珠 (Old)", "legacy/ai_v1_python/main.py"),
    "3": ("五目並べ", "src/main.py"),
    "4": ("五目並べ（強・C++）", "src/cpp_extension/webcam_gomoku_ai.py"),
}

# --- プログラム本体 (ここから下は編集不要です) ---

def clear_screen():
    """コンソール画面をクリアする"""
    os.system('cls' if os.name == 'nt' else 'clear')

def main_hub():
    """モデル選択のハブを実行するメイン関数"""
    
    python_executable = sys.executable

    while True:
        clear_screen() # ★ゲーム終了後、即座にここから実行される
        print("***********************************")
        print("    五目並べ AI セレクションハブ")
        print("***********************************")
        print("\n遊びたいAIモデルの番号を入力してください：\n")
        
        for number, (display_name, filename) in models.items():
            print(f"  [{number}] : {display_name}")
        
        print("\n  [0] : ハブを終了する")
        print("-----------------------------------")
        
        choice = input("番号を入力してください: ").strip()

        if choice == "0":
            print("\nハブを終了します。ありがとうございました。")
            break

        elif choice in models:
            display_name, script_to_run = models[choice]
            print(f"\n...『{display_name}』を起動します ...\n")
            
            try:
                # 選択されたPythonスクリプトを実行
                # ゲームが終了すると、この行の実行が完了します。
                subprocess.run([python_executable, script_to_run], check=True)
                
                # ★★★ 変更点 ★★★
                # ゲーム終了後の確認メッセージとEnterキー入力を削除しました。
                # これにより、ループの最初（clear_screen()）に即座に戻ります。
                #
                # print(f"\n...『{display_name}』が終了しました。")
                # input("Enterキーを押すと、モデル選択に戻ります...")
                #
                # ★★★★★★★★★★★

            except FileNotFoundError:
                print(f"!!! エラー: ファイル '{script_to_run}' が見つかりません。")
                print("!!! `models` 辞書内のファイル名が正しいか確認してください。")
                input("Enterキーを押して続行...") # エラー時は一時停止
            except subprocess.CalledProcessError:
                print(f"!!! エラー: 『{display_name}』の実行中に問題が発生しました。")
                input("Enterキーを押して続行...") # エラー時は一時停止
            except Exception as e:
                print(f"!!! 予期せぬエラーが発生しました: {e}")
                input("Enterキーを押して続行...") # エラー時は一時停止

        else:
            print(f"\n!!! 無効な選択です: '{choice}'")
            print("!!! リストにある番号（0を含む）を入力してください。")
            input("Enterキーを押して続行...") # 選択ミス時は一時停止

if __name__ == "__main__":
    main_hub()