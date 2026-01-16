
import json
import os
import tkinter as tk
from tkinter import ttk, messagebox
import serial.tools.list_ports
import cv2
import subprocess
import sys

CONFIG_FILE = "config.json"

class SettingsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("System Settings")
        self.root.geometry("600x700")

        self.config = self.load_config()

        # Styles
        style = ttk.Style()
        style.configure("TLabel", font=("Arial", 12))
        style.configure("TButton", font=("Arial", 11))

        # --- Gantry Settings ---
        self.create_section("Gantry & Magnet (Arduino)", 0)
        
        self.gantry_port = self.create_port_selector("Gantry Port (Motion):", 
                                                     self.config.get("SERIAL_PORT_GANTRY", "COM9"), 1)
        
        self.magnet_port = self.create_port_selector("Magnet Port (Power):", 
                                                     self.config.get("SERIAL_PORT_MAGNET", "COM10"), 2)

        # --- Coin Sorter Settings ---
        self.create_section("Coin Sorter", 3)
        
        self.sorter_port = self.create_port_selector("Sorter Port:", 
                                                     self.config.get("SERIAL_PORT_SORTER", "COM7"), 4)
        
        self.sorter_cam = self.create_cam_entry("Sorter Camera ID:", 
                                            self.config.get("SORTER_CAM_ID", 2), 5)

        # --- Main Camera Settings ---
        self.create_section("Main Camera (Game AI)", 6)
        
        self.main_cam = self.create_cam_entry("Main Camera ID:", 
                                          self.config.get("CAMERA_INDEX", 0), 7)

        # --- Calibration Buttons ---
        self.create_section("Calibration Tools", 8)
        
        btn_frame = tk.Frame(self.root)
        btn_frame.grid(row=9, column=0, columnspan=2, pady=5, sticky="ew", padx=20)
        
        btn_calib_board = ttk.Button(btn_frame, text="Board (Game) Calibration", command=self.run_board_calibration)
        btn_calib_board.pack(side="left", expand=True, fill="x", padx=5)
        
        btn_calib_sorter = ttk.Button(btn_frame, text="Sorter Calibration", command=self.run_sorter_calibration)
        btn_calib_sorter.pack(side="left", expand=True, fill="x", padx=5)

        # --- Save Button ---
        btn_save = tk.Button(self.root, text="SAVE SETTINGS", bg="#4CAF50", fg="white", 
                             font=("Arial", 14, "bold"), command=self.save_config)
        btn_save.grid(row=10, column=0, columnspan=2, pady=30, sticky="ew", padx=50)

    def create_section(self, title, row):
        lbl = tk.Label(self.root, text=title, font=("Arial", 14, "bold", "underline"), pady=10)
        lbl.grid(row=row, column=0, columnspan=2, sticky="w", padx=10)

    def create_port_selector(self, label_text, default_val, row):
        lbl = ttk.Label(self.root, text=label_text)
        lbl.grid(row=row, column=0, sticky="e", padx=10)
        
        combo = ttk.Combobox(self.root, values=self.get_active_ports(), width=20)
        combo.set(default_val)
        combo.grid(row=row, column=1, sticky="w", padx=10)
        
        # Refresh Button
        btn = ttk.Button(self.root, text="↻", width=3, 
                         command=lambda: combo.config(values=self.get_active_ports()))
        btn.grid(row=row, column=1, sticky="e", padx=(180, 0))
        
        return combo

    def create_cam_entry(self, label_text, default_val, row):
        lbl = ttk.Label(self.root, text=label_text)
        lbl.grid(row=row, column=0, sticky="e", padx=10)
        
        entry = ttk.Entry(self.root, width=23)
        entry.insert(0, str(default_val))
        entry.grid(row=row, column=1, sticky="w", padx=10)
        
        # Preview Button
        btn = ttk.Button(self.root, text="👁 Preview", width=10, 
                         command=lambda: self.preview_camera(entry.get()))
        btn.grid(row=row, column=1, sticky="e", padx=(160, 0))
        
        return entry

    def preview_camera(self, cam_id_str):
        try:
            cam_id = int(cam_id_str)
        except ValueError:
            messagebox.showerror("Error", "Invalid Camera ID")
            return

        print(f"Previewing Camera {cam_id}...")
        
        # Open Camera
        cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        if not cap.isOpened():
             cap = cv2.VideoCapture(cam_id)
        
        if not cap.isOpened():
            messagebox.showerror("Error", f"Could not open Camera {cam_id}")
            return

        win_name = f"Preview - Camera {cam_id} (Pres 'q' or 'ESC' to close)"
        cv2.namedWindow(win_name)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Draw Instruction
            cv2.putText(frame, "Press 'q' or 'ESC' to close", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imshow(win_name, frame)
            
            key = cv2.waitKey(20) & 0xFF
            if key == ord('q') or key == 27 or cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        
        cap.release()
        cv2.destroyAllWindows()

    def get_active_ports(self):
        try:
            ports = serial.tools.list_ports.comports()
            return [p.device for p in ports]
        except:
            return ["Error"]

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_config(self):
        new_config = self.config.copy()
        new_config["SERIAL_PORT_GANTRY"] = self.gantry_port.get()
        new_config["SERIAL_PORT_MAGNET"] = self.magnet_port.get()
        new_config["SERIAL_PORT_SORTER"] = self.sorter_port.get()
        new_config["SORTER_CAM_ID"] = int(self.sorter_cam.get())
        new_config["CAMERA_INDEX"] = int(self.main_cam.get())
        
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(new_config, f, indent=4)
            messagebox.showinfo("Success", "Settings Saved!\nPlease restart applications.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")

    def run_script(self, script_name):
        try:
            # Use absolute path relative to this script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            script_path = os.path.join(script_dir, script_name)
            
            if not os.path.exists(script_path):
                messagebox.showerror("Error", f"File not found: {script_path}")
                return

            subprocess.Popen([sys.executable, script_path])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch {script_name}: {e}")

    def run_board_calibration(self):
        self.run_script("calibrate_board.py")

    def run_sorter_calibration(self):
        self.run_script("calibrate_sorter.py")

if __name__ == "__main__":
    root = tk.Tk()
    app = SettingsApp(root)
    root.mainloop()
