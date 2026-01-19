import tkinter as tk
from tkinter import ttk, scrolledtext
import sys
import os
import threading

# Add path to load modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
try:
    import config
    from hardware.gantry_control import GantryControl
    from hardware.magnet_control import MagnetControl
    from hardware.coin_sorter import CoinSorter
except ImportError as e:
    print(f"Import Error: {e}")
    # Fallback for config
    class config:
        SERIAL_PORT_GANTRY = "COM9"
        SERIAL_PORT_MAGNET = "COM10"
        SERIAL_PORT_SORTER = "COM7"

class ManualControlApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Manual Control - Bridge Project")
        self.root.title("Manual Control - Bridge Project")
        self.root.geometry("800x900")

        # Hardware Interfaces
        self.gantry = GantryControl(config.SERIAL_PORT_GANTRY)
        self.magnet = MagnetControl(config.SERIAL_PORT_MAGNET) # config.SERIAL_PORT_MAGNET should be defined
        self.sorter = CoinSorter(config.SERIAL_PORT_SORTER)

        # UI Layout
        self.create_widgets()

    def create_widgets(self):
        # --- 1. Connection Area ---
        conn_frame = ttk.LabelFrame(self.root, text="Connection")
        conn_frame.pack(fill="x", padx=10, pady=5)

        ttk.Button(conn_frame, text="Connect All", command=self.connect_all).pack(side="left", padx=5, pady=5)
        ttk.Button(conn_frame, text="Disconnect All", command=self.disconnect_all).pack(side="left", padx=5, pady=5)
        
        self.status_lbl = ttk.Label(conn_frame, text="Status: Disconnected", foreground="red")
        self.status_lbl.pack(side="left", padx=10)



        # --- 3. Gantry Control ---
        gan_frame = ttk.LabelFrame(self.root, text="Gantry Control")
        gan_frame.pack(fill="x", padx=10, pady=5)

        # Position (Grid Board)
        pos_frame = ttk.Frame(gan_frame)
        pos_frame.pack(fill="x", pady=5)
        
        # Grid Selection Canvas
        self.selected_grid = (0, 0)
        self.cell_size = 25
        self.board_size = 13
        canvas_size = self.cell_size * self.board_size + 40 # +margin for labels
        
        # Frame for Grid + Button
        grid_container = ttk.Frame(pos_frame)
        grid_container.pack(fill="x", padx=5)

        self.cv_grid = tk.Canvas(grid_container, width=canvas_size, height=canvas_size, bg="white")
        self.cv_grid.pack(side="left", padx=10)
        self.cv_grid.bind("<Button-1>", self.on_grid_click)
        
        # Controls next to grid (Right Side Panel)
        ctrl_panel = ttk.Frame(grid_container)
        ctrl_panel.pack(side="left", fill="both", expand=True, padx=10)
        
        # --- Selected Info & Move ---
        grp_move = ttk.LabelFrame(ctrl_panel, text="Grid Move")
        grp_move.pack(fill="x", pady=5)
        
        self.lbl_selected = ttk.Label(grp_move, text="Selected: (0, 0)", font=("Arial", 12, "bold"))
        self.lbl_selected.pack(pady=5)
        
        ttk.Button(grp_move, text="GO to Selected", command=self.move_grid_selected).pack(fill="x", padx=5, pady=5)

        # --- Utility Buttons (Restored) ---
        grp_util = ttk.LabelFrame(ctrl_panel, text="Quick Actions")
        grp_util.pack(fill="x", pady=5)
        
        ttk.Button(grp_util, text="Go Supply (275, 40)", command=self.go_supply).pack(fill="x", padx=5, pady=2)
        ttk.Button(grp_util, text="Go Zero (0, 0)", command=self.go_zero).pack(fill="x", padx=5, pady=2)
        ttk.Button(grp_util, text="HOME ALL (Calib)", command=self.home_all).pack(fill="x", padx=5, pady=5)

        # --- Z-Axis (Vertical) & Magnet ---
        # Gridレイアウトを使って配置調整
        
        # Magnet Toggle
        self.is_magnet_on = False
        self.btn_magnet = tk.Button(ctrl_panel, text="Magnet OFF", bg="#dddddd", font=("Arial", 10),
                                    command=self.toggle_magnet, height=2)
        self.btn_magnet.pack(fill="x", pady=10)

        # Z-Axis (Vertical)
        grp_z = ttk.LabelFrame(ctrl_panel, text="Z-Axis Control")
        grp_z.pack(fill="x", pady=5)
        
        btn_z_up = ttk.Button(grp_z, text="UP (Home)", command=self.z_up)
        btn_z_up.pack(fill="x", padx=10, pady=5)
        
        btn_z_down = ttk.Button(grp_z, text="DOWN (Pick)", command=self.z_down)
        btn_z_down.pack(fill="x", padx=10, pady=5)


        self.draw_grid()

        # --- 4. Sorter Control ---
        sort_frame = ttk.LabelFrame(self.root, text="Coin Sorter")
        sort_frame.pack(fill="x", padx=10, pady=5)
        
        self.is_sorter_on = False
        self.btn_sorter = tk.Button(sort_frame, text="Sorter OFF", bg="#dddddd", font=("Arial", 10),
                                    command=self.toggle_sorter, height=2)
        self.btn_sorter.pack(fill="x", padx=10, pady=5)
        
        # Speed Sliders
        # Master
        frm_master = ttk.Frame(sort_frame)
        frm_master.pack(fill="x", padx=5, pady=2)
        ttk.Label(frm_master, text="MASTER (All):", width=12).pack(side="left")
        
        self.val_master = tk.StringVar(value="0")
        self.last_master_val = 0 # Track last value
        self.is_updating_master = False # Flag for preventing recursion
        
        self.sca_master = tk.Scale(frm_master, from_=0, to=255, orient="horizontal", command=self.on_master_change)
        self.sca_master.set(0)
        self.sca_master.pack(side="left", fill="x", expand=True)
        ttk.Label(frm_master, textvariable=self.val_master, width=4).pack(side="left")

        ttk.Separator(sort_frame, orient="horizontal").pack(fill="x", padx=10, pady=5)

        # M1 (Feeder)
        frm_m1 = ttk.Frame(sort_frame)
        frm_m1.pack(fill="x", padx=5, pady=2)
        ttk.Label(frm_m1, text="M1 (Feeder):", width=12).pack(side="left")
        self.val_m1 = tk.StringVar(value="0")
        self.sca_m1 = tk.Scale(frm_m1, from_=0, to=255, orient="horizontal", command=self.on_m1_change)
        self.sca_m1.set(0)
        self.sca_m1.pack(side="left", fill="x", expand=True)
        ttk.Label(frm_m1, textvariable=self.val_m1, width=4).pack(side="left")

        # M2 (Conv A)
        frm_m2 = ttk.Frame(sort_frame)
        frm_m2.pack(fill="x", padx=5, pady=2)
        ttk.Label(frm_m2, text="M2 (Conv A):", width=12).pack(side="left")
        self.val_m2 = tk.StringVar(value="0")
        self.sca_m2 = tk.Scale(frm_m2, from_=0, to=255, orient="horizontal", command=self.on_m2_change)
        self.sca_m2.set(0)
        self.sca_m2.pack(side="left", fill="x", expand=True)
        ttk.Label(frm_m2, textvariable=self.val_m2, width=4).pack(side="left")

        # M3 (Stepper) - Max 60 RPM
        frm_m3 = ttk.Frame(sort_frame)
        frm_m3.pack(fill="x", padx=5, pady=2)
        ttk.Label(frm_m3, text="M3 (Step B):", width=12).pack(side="left")
        
        self.val_m3 = tk.StringVar(value="0")
        self.m3_float_val = 0.0 # Maintain float value for smooth scaling

        # Arduino clamps to 60, but let's limit UI to 60 to be safe and clear
        self.sca_m3 = tk.Scale(frm_m3, from_=0, to=60, orient="horizontal", command=self.on_m3_change)
        self.sca_m3.set(0)
        self.sca_m3.pack(side="left", fill="x", expand=True)
        ttk.Label(frm_m3, textvariable=self.val_m3, width=4).pack(side="left")
        ttk.Label(frm_m3, text="RPM", font=("Arial", 8)).pack(side="left")

        # --- Log ---
        log_frame = ttk.LabelFrame(self.root, text="Log")
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.log_area = scrolledtext.ScrolledText(log_frame, height=10)
        self.log_area.pack(fill="both", expand=True)

    def draw_grid(self):
        self.cv_grid.delete("all")
        margin = 30 
        sz = self.cell_size
        
        # Grid Lines
        for i in range(self.board_size):
            # Pos
            pos = margin + i * sz
            end = margin + (self.board_size - 1) * sz
            
            # Label Offset
            label_offset = 15

            # Horizontal Line
            self.cv_grid.create_line(margin, pos, end, pos, fill="black")
            # Label (Y-axis, Left)
            self.cv_grid.create_text(margin - label_offset, pos, text=str(i), font=("Arial", 8))

            # Vertical Line
            self.cv_grid.create_line(pos, margin, pos, end, fill="black")
            # Label (X-axis, Top)
            self.cv_grid.create_text(pos, margin - label_offset, text=str(i), font=("Arial", 8))
            
        # Draw Selected Intersection
        c, r = self.selected_grid
        cx = margin + c * sz
        cy = margin + r * sz
        r_current = 8
        self.cv_grid.create_oval(cx-r_current, cy-r_current, cx+r_current, cy+r_current, fill="red", outline="white")

    def on_grid_click(self, event):
        margin = 30
        sz = self.cell_size
        
        # Snap to nearest intersection
        # x = margin + c * sz  =>  c = (x - margin) / sz
        
        c = round((event.x - margin) / sz)
        r = round((event.y - margin) / sz)
        
        if 0 <= c < self.board_size and 0 <= r < self.board_size:
            self.selected_grid = (c, r)
            self.lbl_selected.config(text=f"Selected: ({c}, {r})")
            self.draw_grid()

    def move_grid_selected(self):
        x, y = self.selected_grid
        threading.Thread(target=self._move_grid_task, args=(x, y)).start()

    def log(self, msg):
        self.log_area.insert(tk.END, msg + "\n")
        self.log_area.see(tk.END)

    # --- Actions ---
    def connect_all(self):
        self.log("Connecting...")
        
        g = self.gantry.connect()
        m = self.magnet.connect()
        # s = self.sorter.connect() # Sorter optional

        if g and m:
            self.status_lbl.config(text="Status: Connected", foreground="green")
            self.log("Gantry & Magnet Connected.")
        else:
            self.status_lbl.config(text="Status: Partial/Failed", foreground="orange")
            self.log(f"Connect Result - Gantry:{g}, Magnet:{m}")

    def disconnect_all(self):
        self.gantry.close()
        self.magnet.close()
        self.sorter.close()
        self.status_lbl.config(text="Status: Disconnected", foreground="red")
        self.log("Disconnected all.")

    def toggle_magnet(self):
        if self.is_magnet_on:
            self.magnet_off()
        else:
            self.magnet_on()

    def magnet_on(self):
        self.magnet.on()
        self.is_magnet_on = True
        self.btn_magnet.config(text="Magnet ON", bg="#ffcccc", foreground="red")
        self.log("Magnet -> ON")

    def magnet_off(self):
        self.magnet.off()
        self.is_magnet_on = False
        self.btn_magnet.config(text="Magnet OFF", bg="#dddddd", foreground="black")
        self.log("Magnet -> OFF")

    def z_down(self):
        threading.Thread(target=self._z_down_task).start()

    def _z_down_task(self):
        self.log("Action: Z Down...")
        res = self.gantry.z_down()
        self.log(f"Result: {res}")

    def z_up(self):
        threading.Thread(target=self._z_up_task).start()

    def _z_up_task(self):
        self.log("Action: Z Up...")
        res = self.gantry.z_up()
        self.log(f"Result: {res}")

    def _move_grid_task(self, x, y):
        self.log(f"Action: Move to Grid({x}, {y})...")
        res = self.gantry.move_to_grid(x, y)
        self.log(f"Result: {res}")

    def go_supply(self):
        threading.Thread(target=lambda: self.log(f"Supply Result: {self.gantry.move_to_supply()}")).start()

    def go_zero(self):
        threading.Thread(target=lambda: self.log(f"Zero Result: {self.gantry.move_to_zero()}")).start()

    def home_all(self):
        if tk.messagebox.askyesno("Confirm", "Re-run HOMING sequence?"):
            threading.Thread(target=lambda: self.log(f"Homing Result: {self.gantry.home_all()}")).start()

    def toggle_sorter(self):
        if self.is_sorter_on:
            self.sorter_stop()
        else:
            self.sorter_start()

    # --- Slider Callbacks ---
    def on_master_change(self, val):
        self.is_updating_master = True # Flag start
        try:
            new_v = int(val)
            delta = new_v - self.last_master_val
            self.last_master_val = new_v
            
            self.val_master.set(str(new_v))
            
            # Calculate Delta for M3 (Scaled 60/255 approx 0.235)
            delta_m3 = delta * (60.0 / 255.0)
            
            # Accumulate float value
            self.m3_float_val += delta_m3
            self.m3_float_val = max(0.0, min(60.0, self.m3_float_val))

            # Apply delta for M1, M2
            try:
                m1 = int(self.sca_m1.get()) + delta
                m2 = int(self.sca_m2.get()) + delta
            except ValueError:
                m1 = 0
                m2 = 0

            m1 = max(0, min(255, m1))
            m2 = max(0, min(255, m2))
            
            # Use accumulated float val for M3
            m3_int = int(round(self.m3_float_val))
            
            # Update UI sliders
            # NOTE: These .set() calls might trigger on_mX_change events depending on implementation.
            # The flag is_updating_master prevents recursion/override.
            self.sca_m1.set(m1)
            self.sca_m2.set(m2)
            self.sca_m3.set(m3_int)
            
            # Update display vars
            self.val_m1.set(str(m1))
            self.val_m2.set(str(m2))
            self.val_m3.set(str(m3_int))
            
            # Send commands
            self.send_speed_m1(m1)
            self.send_speed_m2(m2)
            self.send_speed_m3(m3_int)
            
        finally:
            self.is_updating_master = False # Release flag

    def on_m1_change(self, val):
        if hasattr(self, 'is_updating_master') and self.is_updating_master: return
        v = int(val)
        self.val_m1.set(str(v))
        self.send_speed_m1(v)

    def on_m2_change(self, val):
        if hasattr(self, 'is_updating_master') and self.is_updating_master: return
        v = int(val)
        self.val_m2.set(str(v))
        self.send_speed_m2(v)

    def on_m3_change(self, val):
        if hasattr(self, 'is_updating_master') and self.is_updating_master: return
        v = int(val)
        # Sync float val when manually changed
        self.m3_float_val = float(v)
        self.val_m3.set(str(v))
        self.send_speed_m3(v)

    # --- Send Helpers ---
    def send_speed_m1(self, spd):
        if self.is_sorter_on: self.sorter.set_feeder_speed(spd)

    def send_speed_m2(self, spd):
        if self.is_sorter_on: self.sorter.set_conveyor_a_speed(spd)

    def send_speed_m3(self, spd):
        if self.is_sorter_on: self.sorter.set_stepper_speed(spd)

    def sorter_start(self):
        self.sorter.connect()
        # Send current slider values
        self.sorter.set_feeder_speed(self.sca_m1.get())
        self.sorter.set_conveyor_a_speed(self.sca_m2.get())
        self.sorter.set_stepper_speed(self.sca_m3.get())
        
        self.is_sorter_on = True
        self.btn_sorter.config(text="Sorter ON", bg="#ccffcc", foreground="green")
        self.log("Sorter START")

    def sorter_stop(self):
        self.sorter.stop_all()
        self.sorter.close()
        self.is_sorter_on = False
        self.btn_sorter.config(text="Sorter OFF", bg="#dddddd", foreground="black")
        self.log("Sorter STOP")

if __name__ == "__main__":
    import tkinter.messagebox
    root = tk.Tk()
    app = ManualControlApp(root)
    root.mainloop()
