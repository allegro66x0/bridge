
import cv2
import numpy as np
import json
import os
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

# Settings
CAM_INDEX = config.CAMERA_INDEX
CONFIG_FILE = "config.json"

clicked_points = []
img_static = None
img_display = None

def mouse_callback(event, x, y, flags, param):
    global clicked_points, img_display
    if event == cv2.EVENT_LBUTTONDOWN:
        # Display is 2x scale
        real_x, real_y = x // 2, y // 2
        
        if len(clicked_points) < 4:
            clicked_points.append((real_x, real_y))
            print(f"Verified Point {len(clicked_points)}: ({real_x}, {real_y})")

def update_config_file(points):
    """Update config.json with new points"""
    if not os.path.exists(CONFIG_FILE):
        print(f"Error: {CONFIG_FILE} not found.")
        return

    try:
        with open(CONFIG_FILE, 'r') as f:
            data = json.load(f)
        
        data["FIXED_CORNER_POINTS"] = points
        
        with open(CONFIG_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f"✅ Config updated! FIXED_CORNER_POINTS = {points}")

    except Exception as e:
        print(f"❌ Failed to update config: {e}")

def main():
    global img_static, img_display, clicked_points
    
    # 1. Capture Static Image
    print(f"Opening Camera {CAM_INDEX}...")
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAM_INDEX)
    
    if not cap.isOpened():
        print(f"Error: Could not open Camera {CAM_INDEX}")
        return

    # Warmup
    for _ in range(10): cap.read()
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Could not capture frame.")
        return

    print("✅ Frame Captured. Closing camera to freeze image.")
    img_static = frame.copy()

    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", mouse_callback)

    print("="*50)
    print("【Board Calibration】(Static Image Mode)")
    print("Click 4 corners: Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left")
    print("Press 's' to SAVE and Exit.")
    print("Press 'r' to RESET points.")
    print("Press 'q' to QUIT without saving.")
    print("="*50)

    while True:
        # Refresh display image from static base
        h, w = img_static.shape[:2]
        img_display = cv2.resize(img_static, (w*2, h*2)) # 2x Zoom

        # Draw Points
        for i, pt in enumerate(clicked_points):
            disp_pt = (pt[0]*2, pt[1]*2)
            cv2.circle(img_display, disp_pt, 10, (0, 0, 255), -1)
            cv2.putText(img_display, str(i+1), (disp_pt[0]+20, disp_pt[1]-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # Draw Polygon if complete
        if len(clicked_points) == 4:
            pts = np.array([(p[0]*2, p[1]*2) for p in clicked_points], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img_display, [pts], True, (0, 255, 0), 3)
            
            cv2.putText(img_display, "Press 's' to SAVE", (40, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)

        cv2.imshow("Calibration", img_display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('s'):
            if len(clicked_points) == 4:
                print("Saving to config.json...")
                update_config_file(clicked_points)
                break
            else:
                print(f"Need 4 points. You have {len(clicked_points)}.")
        
        elif key == ord('r'):
            clicked_points = []
            print("Points Reset.")

        elif key == ord('q'):
            print("Quit without saving.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
