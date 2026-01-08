import tkinter as tk
from tkinter import messagebox
import subprocess
import sys
import os

# --- ユーザー設定エリア ---
models = {
    "1": ("Level6", "L6/webcam_gomoku_ai.py"),
    "2": ("Level5", "L5/main5.py"),
    "3": ("Level4", "L4/main4.py"),
    "4": ("Level3", "L3/main3.py"),
    "5": ("Level2", "L2/main2.py"),
    "6": ("Level1", "L1/main1.py"),
}

class GomokuLauncherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("五目並べ AI セレクションハブ")
        self.root.attributes('-fullscreen', True)
        
        # 背景色を少しスタイリッシュに
        self.root.configure(bg="#f0f0f0")

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        title_label = tk.Label(
            root, 
            text="対戦するAIを選んでください", 
            font=("Meiryo UI", 30, "bold"),
            bg="#f0f0f0",
            pady=screen_height * 0.05
        )
        title_label.pack()

        button_frame = tk.Frame(root, bg="#f0f0f0")
        button_frame.pack(expand=True, fill="both", padx=150, pady=20)

        for key, (display_name, script_path) in models.items():
            btn = tk.Button(
                button_frame, 
                text=f"{display_name}", 
                font=("Arial", 28),
                height=1, 
                bg="#ffffff",
                relief="flat", # フラットデザイン風
                borderwidth=1,
                command=lambda p=script_path, n=display_name: self.run_ai_model(p, n)
            )
            btn.pack(fill="x", pady=10)

        exit_btn = tk.Button(
            root, 
            text="終了する", 
            font=("Meiryo UI", 24), 
            bg="#ffcccc", 
            height=2,
            relief="flat",
            command=root.destroy
        )
        exit_btn.pack(fill="x", padx=150, pady=50)

    def run_ai_model(self, script_path, display_name):
        """選択されたPythonスクリプトを実行する（黒画面待機版）"""
        python_executable = sys.executable
        
        if not os.path.exists(script_path):
            messagebox.showerror("エラー", f"ファイルが見つかりません:\n{script_path}", parent=self.root)
            return

        # ★ここがポイント：画面全体を黒いフレームで覆う
        cover_frame = tk.Frame(self.root, bg="black")
        cover_frame.place(relx=0, rely=0, relwidth=1, relheight=1)
        
        # 画面を強制更新して真っ黒にする
        self.root.update()

        try:
            # ★subprocess.run を使い、ゲームが終わるまで待つ
            subprocess.run([python_executable, script_path])
        except Exception as e:
            messagebox.showerror("実行エラー", f"起動中にエラーが発生しました:\n{e}", parent=self.root)
        finally:
            # ゲームが終わったら黒いカバーを外す
            cover_frame.destroy()

# --- メイン処理 ---
if __name__ == "__main__":
    root = tk.Tk()
    app = GomokuLauncherApp(root)
    root.mainloop()