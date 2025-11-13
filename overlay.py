import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)  # your webcam
if not cap.isOpened():
    print("Could not open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    # Draw boxes directly (Ultralytics[YOLO] handles overlay)
    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Overlay Test", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
