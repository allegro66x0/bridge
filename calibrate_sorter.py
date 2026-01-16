
import cv2
import json
import os
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

ROI_FILE = config.SORTER_ROI_PATH
CAM_ID = config.SORTER_CAM_ID

def main():
    print(f"Opening Camera {CAM_ID} for Sorter Calibration...")
    cap = cv2.VideoCapture(CAM_ID, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAM_ID)
        if not cap.isOpened():
            print(f"Error: Could not open camera {CAM_ID}")
            return

    # Read one frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        return

    print("Please select the area for the Coin Sorter.")
    print("Draw a rectangle and press ENTER or SPACE to confirm.")
    print("Press c to cancel.")
    
    # Select ROI
    # fromCenter=False, showCrosshair=True
    r = cv2.selectROI("Sorter Calibration", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    cap.release()

    # r = (x, y, w, h)
    if r[2] > 0 and r[3] > 0:
        roi_data = {"x": int(r[0]), "y": int(r[1]), "w": int(r[2]), "h": int(r[3])}
        
        # Save to JSON
        with open(ROI_FILE, 'w') as f:
            json.dump(roi_data, f, indent=4)
        
        print(f"✅ Sorter ROI Saved to {ROI_FILE}: {roi_data}")
        
        # Show confirmation (Optional, just print is fine)
    else:
        print("Calibration Cancelled.")

if __name__ == "__main__":
    main()
