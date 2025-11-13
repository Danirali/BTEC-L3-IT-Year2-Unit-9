import cv2
import time

CAPTURE_DELAY = 3

# Replace this with your RTSP URL
rtsp_url = "rtsp://USER:PASS@IP_ADDRESS:PORT/STREAM_PATH"

cap = cv2.VideoCapture(rtsp_url)
img_count = 0

if not cap.isOpened():
    print("Error: Cannot open the RTSP stream.")
else:
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from RTSP stream")
                break
            filename = f"capture_{img_count}.jpg"
            cv2.imwrite(filename, frame)
            img_count += 1
            time.sleep(CAPTURE_DELAY)  # Wait 3 seconds
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
