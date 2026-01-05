import cv2

for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✅ Camera index {i} is working")
        else:
            print(f"⚠️ Camera index {i} opened but failed to read")
        cap.release()
    else:
        print(f"❌ Camera index {i} not available")