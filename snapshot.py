import cv2
import time

CAPTURE_DELAY = 3

# Open the default camera (0)
cap = cv2.VideoCapture(0)

img_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        # Save the image
        filename = f"capture_{img_count}.jpg"
        cv2.imwrite(filename, frame)
        img_count += 1
        # Wait for 3 seconds
        time.sleep(CAPTURE_DELAY)
except KeyboardInterrupt:
    print("Stopped by user")
finally:
    cap.release()
    cv2.destroyAllWindows()
