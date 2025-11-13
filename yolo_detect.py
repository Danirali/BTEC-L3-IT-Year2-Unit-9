from ultralytics import YOLO
import cv2

# Load a pretrained YOLOv8 model
model = YOLO('yolov8n.pt')  # Use yolov8s.pt, yolov8m.pt etc. based on needs

# Load image
def scan(image):
    if not image:
        for i in range(1, 4):
            image_path = f'capture_{i}.jpg'
            img = cv2.imread(image_path)

            # Run inference
            results = model(img)

            # Visualize results on the image
            annotated_img = results[0].plot()  # Plots all detections

            cv2.imshow("YOLOv8 Person Detection", annotated_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    img = cv2.imread(image)

    # Run inference
    results = model(img)

    # Visualize results on the image
    annotated_img = results[0].plot()  # Plots all detections

    cv2.imshow("YOLOv8 Person Detection", annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

scan('')
# scan('test.jpg')