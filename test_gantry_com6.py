import serial
import time

PORT = 'COM6'
BAUDRATE = 9600

def main():
    print(f"Connecting to {PORT}...")
    try:
        ser = serial.Serial(PORT, BAUDRATE, timeout=2)
        time.sleep(2) # Wait for reset
        print("Connected.")
        
        # Clear buffer
        ser.reset_input_buffer()

        print("Sending '0000' (Home Command)...")
        ser.write(b"0000\n")
        
        # Read response
        start = time.time()
        while time.time() - start < 5:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                print(f"Received: {line}")
                if line == "READY":
                    print("-> Success! Gantry is ready.")
                    break
            time.sleep(0.1)
            
        ser.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
