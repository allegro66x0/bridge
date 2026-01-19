import serial
import time
import threading

class GantryControl:
    def __init__(self, port, baud_rate=9600):
        self.port = port
        self.baud_rate = baud_rate
        self.ser = None
        self.is_connected = False
        self.lock = threading.Lock()

    def connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baud_rate, timeout=1)
            time.sleep(2) # Arduino Reset wait
            self.is_connected = True
            print(f"✅ Gantry Connected: {self.port}")
            return True
        except serial.SerialException as e:
            print(f"❌ Gantry Connection Failed: {e}")
            self.is_connected = False
            return False

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.is_connected = False
            print("Gantry Disconnected")

    def _send_cmd(self, cmd):
        if not self.ser or not self.is_connected:
            return "Error: Not Connected"
        
        with self.lock:
            try:
                full_cmd = f"{cmd}\n"
                print(f"-> Gantry: {cmd}")
                self.ser.write(full_cmd.encode('utf-8'))
                
                # Wait for READY
                while True:
                    if self.ser.in_waiting > 0:
                        line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                        if line == "READY":
                            return "READY"
                        # print(f"  <- {line}")
            except Exception as e:
                print(f"Send Error: {e}")
                return str(e)

    def move_to_grid(self, x, y):
        # AI Grid (0-12) to Gantry Grid (1-13)
        gx = x + 1
        gy = y + 1
        return self._send_cmd(f"M:{gx:02}{gy:02}")

    def move_to_supply(self):
        return self._send_cmd("M:SPLY")
    
    def move_to_zero(self):
        return self._send_cmd("M:ZERO")

    def z_down(self):
        return self._send_cmd("Z:PICK")

    def z_up(self):
        return self._send_cmd("Z:HOME")

    def home_all(self):
        return self._send_cmd("H:ALL")
