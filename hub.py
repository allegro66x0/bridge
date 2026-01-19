import tkinter as tk
from tkinter import messagebox
import subprocess
import sys
import os
import threading
import queue
import time
import json
import serial.tools.list_ports
import cv2

# --- ユーザー設定エリア ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE_CACHE_FILE = os.path.join(BASE_DIR, "device_cache.json")

models = {
    "1": ("Gomoku AI (Final)", os.path.join(BASE_DIR, "L6", "webcam_gomoku_ai.py")),
    "2": ("Manual Control", os.path.join(BASE_DIR, "manual_control_ui.py")),
}

class LoadingApp:
    def __init__(self, root, on_complete_callback):
        self.root = root
        self.on_complete_callback = on_complete_callback
        self.root.title("System Initialization")
        self.root.geometry("800x600")
        self.root.configure(bg="#000000")
        
        # UI Elements
        self.lbl_title = tk.Label(root, text="SYSTEM INITIALIZING...", font=("Consolas", 24, "bold"), fg="#00ff00", bg="#000000")
        self.lbl_title.pack(pady=20)
        
        self.txt_log = tk.Text(root, bg="#101010", fg="#00ff00", font=("Consolas", 10), height=25, width=90)
        self.txt_log.pack(padx=20, pady=10)
        
        self.msg_queue = queue.Queue()
        self.running = True
        
        # Start Scanning Thread
        self.thread = threading.Thread(target=self.run_scans, daemon=True)
        self.thread.start()
        
        # Start Log Polling
        self.process_queue()

    def log(self, message):
        self.msg_queue.put(message)

    def process_queue(self):
        if not self.running: return
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                self.txt_log.insert(tk.END, f"> {msg}\n")
                self.txt_log.see(tk.END)
                if msg == "--- COMPLETE ---":
                    self.finish_loading()
                    return
        except queue.Empty:
            pass
        self.root.after(100, self.process_queue)

    def run_scans(self):
        time.sleep(1) # Wait for UI
        self.log("Starting System Diagnostics...")
        
        scan_results = {
            "cameras": [],
            "serial_ports": []
        }
        
        # 1. Serial Ports
        self.log("Scanning Serial Ports...")
        try:
            ports = serial.tools.list_ports.comports()
            for p in ports:
                self.log(f"  [FOUND] {p.device} - {p.description}")
                scan_results["serial_ports"].append(p.device)
            if not ports:
                self.log("  [WARN] No serial ports found.")
        except Exception as e:
            self.log(f"  [ERROR] Serial scan failed: {e}")

        # 2. Cameras
        self.log("Scanning Cameras (0-9)...")
        # Reuse logic similar to settings_ui but simplified
        for i in range(10):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    self.log(f"  [FOUND] Camera ID {i}")
                    scan_results["cameras"].append(i)
                    cap.release()
                else:
                    # Fallback try
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        self.log(f"  [FOUND] Camera ID {i}")
                        scan_results["cameras"].append(i)
                        cap.release()
            except Exception as e:
                pass # Ignore errors during scan
        
        # 3. Save Cache
        self.log("Saving Device Cache...")
        try:
            with open(DEVICE_CACHE_FILE, 'w') as f:
                json.dump(scan_results, f, indent=4)
            self.log(f"  Saved to {DEVICE_CACHE_FILE}")
        except Exception as e:
            self.log(f"  [ERROR] Failed to save cache: {e}")

        # 4. Check Scripts
        self.log("Checking AI Models...")
        for k, (name, path) in models.items():
            if os.path.exists(path):
                self.log(f"  [OK] {name}")
            else:
                self.log(f"  [MISSING] {path}")

        self.log("Initialization Complete.")
        time.sleep(1) # Let user see the message
        self.log("--- COMPLETE ---")

    def finish_loading(self):
        self.running = False
        self.root.destroy()
        self.on_complete_callback()

class GomokuLauncherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("五目並べ AI セレクションハブ")
        # self.root.attributes('-fullscreen', True) # フルスクリーン無効化 (安定性のため)
        self.root.geometry("1024x768")
        
        # 背景色を少しスタイリッシュに
        self.root.configure(bg="#f0f0f0")

        # screen_width = self.root.winfo_screenwidth()
        # screen_height = self.root.winfo_screenheight()
        screen_width = 1024
        screen_height = 768

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

        # Settings Button (Top-Right, Square)
        self.settings_btn = tk.Button(
            root,
            text="⚙️",
            font=("Meiryo UI", 20),
            bg="#e0e0e0",
            relief="flat",
            command=self.open_settings
        )
        # Place in top-right corner
        self.settings_btn.place(relx=0.95, rely=0.05, anchor="ne", width=80, height=80)
        self.settings_btn.lift() # 最前面に表示

        exit_btn = tk.Button(
            root, 
            text="終了する", 
            font=("Meiryo UI", 24), 
            bg="#ffcccc", 
            height=2,
            relief="flat",
            command=root.destroy
        )
        exit_btn.pack(fill="x", padx=150, pady=20)

    def open_settings(self):
        """設定画面を開く"""
        try:
            # hub.pyと同じ場所にある settings_ui.py を探す
            script_dir = os.path.dirname(os.path.abspath(__file__))
            settings_path = os.path.join(script_dir, "settings_ui.py")
            
            if not os.path.exists(settings_path):
                messagebox.showerror("Error", f"Settings UI not found:\n{settings_path}")
                return

            subprocess.Popen([sys.executable, settings_path])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch Settings:\n{e}")

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
def launch_main_menu():
    root = tk.Tk()
    app = GomokuLauncherApp(root)
    root.mainloop()

if __name__ == "__main__":
    # Phase 1: Loading Screen
    load_root = tk.Tk()
    loading_app = LoadingApp(load_root, on_complete_callback=launch_main_menu)
    load_root.mainloop()