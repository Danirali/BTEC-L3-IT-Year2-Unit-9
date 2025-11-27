import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open camera")
    exit()

# Initialize heatmap storage (will be created once we know frame size)
heatmap = None

# Heatmap parameters
DECAY = 0.98   # >1 = no decay, <1 = slowly fades
INTENSITY = 40  # How much to increase per detection point

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # Create heatmap on first frame
    if heatmap is None:
        heatmap = np.zeros((h, w), dtype=np.float32)

    # Run YOLO inference
    results = model(frame, verbose=False)
    detections = results[0].boxes

    # Update heatmap based on person detections only (class 0)
    if detections is not None and len(detections) > 0:
        for box in detections:
            cls = int(box.cls[0])

            if cls == 0:  # class 0 = person
                x1, y1, x2, y2 = box.xyxy[0]
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Add intensity at person centroid
                if 0 <= cx < w and 0 <= cy < h:
                    heatmap[cy, cx] += INTENSITY

    # Apply heatmap decay each frame
    heatmap *= DECAY

    # Normalize heatmap for coloring
    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_norm = heatmap_norm.astype(np.uint8)

    # Apply OpenCV colormap: Blue → Red
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

    # Blend heatmap with original frame (0.6 = transparency)
    overlay = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)

    # Draw YOLO boxes
    annotated_frame = results[0].plot()

    # Combine YOLO detections with the heatmap
    final_output = cv2.addWeighted(annotated_frame, 0.7, heatmap_color, 0.3, 0)

    cv2.imshow("YOLO + Heatmap", final_output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
