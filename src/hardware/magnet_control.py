
import serial
import time

class MagnetControl:
    def __init__(self, port, baud_rate=9600):
        self.port = port
        self.baud_rate = baud_rate
        self.ser = None
        self.is_connected = False

    def connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baud_rate, timeout=1)
            time.sleep(2) # Arduino Reset wait
            self.is_connected = True
            print(f"✅ Magnet Connected: {self.port}")
            return True
        except serial.SerialException as e:
            print(f"❌ Magnet Connection Failed: {e}")
            self.is_connected = False
            return False

    def on(self):
        if self.ser and self.is_connected:
            self.ser.write(b'1')
            # レスポンス待機するならここでreadline
            # time.sleep(0.1) 
            print("🧲 Magnet ON")

    def off(self):
        if self.ser and self.is_connected:
            self.ser.write(b'0')
            print("🧲 Magnet OFF")

    def close(self):
        if self.ser and self.ser.is_open:
            self.off()
            self.ser.close()
            self.is_connected = False
