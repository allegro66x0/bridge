import cv2
import os
import json
import time

# Load Configuration
CONFIG_PATH = "config.json"
DATA_DIR_0 = "../../data/o"
DATA_DIR_1 = "../../data/1"

def main():
    # Load config to get camera ID
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
            cam_id = config["system_a"]["cam_id"]
    else:
        cam_id = 0
        print("Config not found, defaulting to Cam ID 0")

    # Create directories if not exist
    os.makedirs(DATA_DIR_0, exist_ok=True)
    os.makedirs(DATA_DIR_1, exist_ok=True)
    
    # Load ROI if exists
    roi = None
    roi_path = "model_roi.json"
    if os.path.exists(roi_path):
        with open(roi_path, 'r') as f:
            roi = json.load(f)
            print(f"Loaded ROI: {roi}")

    print(f"Attempting to open Camera {cam_id} using DSHOW...")
    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"DSHOW Failed. Trying Default Backend for ID {cam_id}...")
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            print("Camera failed. Trying ID 0 (Backup)...")
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    print("--- Data Collection Mode ---")
    print("Commands:")
    print(" '0': Save as EMPTY (Folder 'o')")
    print(" '1': Save as COIN  (Folder '1')")
    print(" 'q': Quit")
    
    count_0 = len(os.listdir(DATA_DIR_0))
    count_1 = len(os.listdir(DATA_DIR_1))
    print(f"Current Count -> Empty: {count_0}, Coin: {count_1}")

    while True:
        ret, frame = cap.read()
        if not ret: break

        display_frame = frame.copy()
        
        # Draw ROI
        if roi:
            x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        
        cv2.imshow("Data Collector", display_frame)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('0'):
            # Save to 0
            ts = int(time.time() * 1000)
            fname = f"capture_{ts}_Pro.jpg"
            path = os.path.join(DATA_DIR_0, fname)
            cv2.imwrite(path, frame)
            print(f"Saved EMPTY: {fname}")
            count_0 += 1
        elif key == ord('1'):
            # Save to 1
            ts = int(time.time() * 1000)
            fname = f"capture_{ts}_Pro.jpg"
            path = os.path.join(DATA_DIR_1, fname)
            cv2.imwrite(path, frame)
            print(f"Saved COIN: {fname}")
            count_1 += 1
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
