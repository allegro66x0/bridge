import serial
import time
import sys

def check_port(port):
    try:
        ser = serial.Serial(port, 9600, timeout=3)
        print(f"Checking {port}...")
        
        # Probe 1: Wait for startup
        time.sleep(2) 
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            print(f"  [{port}] Startup: {line}")
            if "Coin Sorter" in line: return "SORTER"
            if line == "READY": return "GANTRY"
            
        # Probe 2: Send Gantry Command (0000) - harmless home check? 
        # Actually 0000 moves it. Maybe just send newline? 
        # Or send a dummy char?
        # If I send "9999" (invalid), Gantry says "ERROR", Sorter says nothing (buffer check)
        
        ser.write(b"\n")
        time.sleep(0.5)
        if ser.in_waiting > 0:
             line = ser.readline().decode('utf-8', errors='ignore').strip()
             print(f"  [{port}] Response to NL: {line}")
             if "READY" in line: return "GANTRY" # Gantry repeats READY on newline? No code doesn't say that.
             
        # Probe 3: Send "STOP" (Sorter safe command)
        ser.write(b"STOP\n")
        time.sleep(0.5)
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            print(f"  [{port}] Response to STOP: {line}")
            if "STOPPED" in line: return "SORTER"

        ser.close()
        return "UNKNOWN"
    except serial.SerialException:
        return "ERROR"
    except Exception as e:
        print(f"Exception: {e}")
        return "ERROR"

def main():
    detected = {}
    candidates = ['COM4', 'COM5', 'COM6']
    
    for port in candidates:
        result = check_port(port)
        if result != "ERROR":
            detected[port] = result
            print(f"-> {port} is {result}")
    
    print("\nSummary:")
    for p, r in detected.items():
        print(f"{p}: {r}")

if __name__ == "__main__":
    main()
