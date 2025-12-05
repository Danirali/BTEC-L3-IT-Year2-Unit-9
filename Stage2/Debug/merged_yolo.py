import cv2
import numpy as np
import json
import os
from ultralytics import YOLO

# -----------------------------
# CONFIGURATION
# -----------------------------
MODEL_PATH = "yolov8n.pt"
INTENSITY = 70     # heatmap blob intensity
RADIUS = 50        # blob radius
DECAY_MINUTES = 60 # heatmap fade time
DEFAULT_FPS = 30
# -----------------------------

model = YOLO(MODEL_PATH)


# -----------------------------
# Save Functions
# -----------------------------

def save_heatmap(x, y, point_intensity):
    with open('heatmap.txt', 'a') as f:
        f.write(json.dumps({"x": x, "y": y, "intensity": point_intensity}) + ",")

def save_raw(heatmap, x, y, y_min, y_max, x_min, x_max, intensity):
    with open('raw.txt', 'a') as f:
        f.write(json.dumps({
            "heatmap": heatmap.tolist(),
            "x": x, "y": y,
            "y_min": y_min, "y_max": y_max,
            "x_min": x_min, "x_max": x_max,
            "intensity": intensity
        }) + ",")


# -----------------------------
# Heatmap Blob Function
# -----------------------------

def add_blob(heatmap, x, y, radius=RADIUS, intensity=INTENSITY):
    h, w = heatmap.shape
    y_min = max(0, y - radius)
    y_max = min(h, y + radius)
    x_min = max(0, x - radius)
    x_max = min(w, x + radius)

    for j in range(y_min, y_max):
        for i in range(x_min, x_max):
            if (i - x)**2 + (j - y)**2 <= radius**2:
                heatmap[j, i] += intensity
                blob_intensity = heatmap[j, i]

    save_heatmap(x, y, str(blob_intensity))
    save_raw(heatmap, x, y, y_min, y_max, x_min, x_max, blob_intensity)


# -----------------------------
# YOLO + HEATMAP PROCESSOR
# -----------------------------

def process_frame(frame, heatmap, decay_factor):
    h, w = frame.shape[:2]

    # Create heatmap on first frame
    if heatmap is None:
        heatmap = np.zeros((h, w), dtype=np.float32)

    # YOLO inference
    results = model(frame, verbose=False)
    detections = results[0].boxes

    # Update heatmap for PEOPLE only (class 0)
    if detections is not None:
        for box in detections:
            cls = int(box.cls[0])
            if cls == 0:  # person
                x1, y1, x2, y2 = box.xyxy[0]
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                if 0 <= cx < w and 0 <= cy < h:
                    add_blob(heatmap, cx, cy)

    # Decay heatmap
    heatmap *= decay_factor

    # Normalized heatmap → color map
    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

    # Draw YOLO boxes
    annotated_frame = results[0].plot()

    # Blend with heatmap
    final_image = cv2.addWeighted(annotated_frame, 0.7, heatmap_color, 0.3, 0)

    return final_image, heatmap


# -----------------------------
# RUN FOR WEBCAM, VIDEO, OR IMAGES
# -----------------------------

def run(input_path=""):
    heatmap = None

    if input_path == "":
        print("➡ Using webcam...")
        cap = cv2.VideoCapture(0)
        fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS

        decay_factor = 0.5 ** (1 / (fps * DECAY_MINUTES * 60))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            output, heatmap = process_frame(frame, heatmap, decay_factor)
            cv2.imshow("YOLO + Heatmap", output)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return

    # -----------------------------
    # If FILE Provided: Image or Video
    # -----------------------------

    if os.path.isdir(input_path):
        print("➡ Processing folder of images...")
        for f in os.listdir(input_path):
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                img = cv2.imread(os.path.join(input_path, f))
                output, heatmap = process_frame(img, heatmap, 1)  # no decay for images
                cv2.imshow("YOLO + Heatmap", output)
                cv2.waitKey(0)

        cv2.destroyAllWindows()
        return

    if input_path.lower().endswith((".mp4", ".mov", ".avi")):
        print("➡ Processing video...")
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS

        decay_factor = 0.5 ** (1 / (fps * DECAY_MINUTES * 60))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            output, heatmap = process_frame(frame, heatmap, decay_factor)
            cv2.imshow("YOLO + Heatmap", output)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return

    # Single image
    print("➡ Processing single image...")
    img = cv2.imread(input_path)
    output, _ = process_frame(img, heatmap, 1)
    cv2.imshow("YOLO + Heatmap", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# -----------------------------
# RUN IT
# -----------------------------
run("")  
# run("test.jpg")
# run("my_video.mp4")
# run("captures_folder/")
