from ultralytics import YOLO
import cv2

# Load a pretrained YOLOv8 model
model = YOLO('yolov8n.pt')  # Use yolov8s.pt, yolov8m.pt etc. based on needs

# Load image
for i in range(1, 5):
    image_path = f'capture_{i}.jpg'
    img = cv2.imread(image_path)

    # Run inference
    results = model(img)

    # Visualize results on the image
    annotated_img = results[0].plot()  # Plots all detections

    cv2.imshow("YOLOv8 Person Detection", annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
