import cv2
import numpy as np
import pandas as pd
import time
from ultralytics import YOLO

# 1. SETUP & INITIALIZATION
# Load the YOLOv8 model (Nano version for speed in demo)
model = YOLO("yolov8n.pt")

# Load the "Empty Store" image for the Dashboard Heatmap
# If you don't have this yet, it will create a blank one from the first frame
store_bg = cv2.imread("store_layout.jpg") 

# Data Storage
track_history = {} # Stores {id: {'start': float, 'hits': int}}
analytics_log = []

# Heatmap parameters
INTENSITY = 2 # Gradual build-up per frame
DECAY = 0.995 # Slow fade for real-time visualization

def get_zone(x, width): # Removed 'y' from here
    """Classifies coordinates into store zones based on horizontal position."""
    if x < width / 3:
        return "Entrance"
    elif x > (2 * width / 3):
        return "Checkout"
    else:
        return "Aisle_1"

def calculate_engagement(dwell, hits):
    """Beta Engagement Formula: (DwellTime * 0.5) + (Detection_Frames * 0.2)"""
    return round((dwell * 0.5) + (hits * 0.2), 2)

# 2. VIDEO CAPTURE
cap = cv2.VideoCapture(0) # Use 0 for Webcam or 'video.mp4' for file

# Initialize heatmap array once resolution is known
heatmap_data = None

print("Footfall Tracker Beta Started. Press 'q' to stop and save data.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    h, w, _ = frame.shape
    if heatmap_data is None:
        heatmap_data = np.zeros((h, w), dtype=np.float32)
        if store_bg is None: # Fallback if bg image missing
            store_bg = np.zeros_like(frame)

    # 3. AI DETECTION & TRACKING
    # persist=True ensures IDs stay the same across frames
    results = model.track(frame, persist=True, verbose=False, classes=[0])

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        ids = results[0].boxes.id.int().cpu().tolist()
        confs = results[0].boxes.conf.float().cpu().tolist()

        for box, track_id, conf in zip(boxes, ids, confs):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Update Tracking History
            if track_id not in track_history:
                track_history[track_id] = {'start': time.time(), 'hits': 1}
            else:
                track_history[track_id]['hits'] += 1

            # Update Heatmap (Logic for the Dashboard)
            cv2.circle(heatmap_data, (cx, cy), 30, (INTENSITY), -1)

            # 4. BIG DATA LOGGING (For Unit 10 Task 4)
            dwell_time = time.time() - track_history[track_id]['start']
            
            analytics_log.append({
                "DetectionID": track_id,
                "Timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                "Zone": get_zone(cx, w),
                "X_Coord": cx,
                "Y_Coord": cy,
                "DwellTime_s": round(dwell_time, 2),
                "EngagementScore": calculate_engagement(dwell_time, track_history[track_id]['hits']),
                "AI_Confidence": round(conf, 2)
            })

    # 5. VISUALIZATION (Dashboard View)
    # Apply decay to heatmap
    heatmap_data *= DECAY
    
    # Normalize and color the heatmap
    heatmap_norm = cv2.normalize(heatmap_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

    # Blend Heatmap with the STILL STORE IMAGE (Privacy Mode)
    dashboard_overlay = cv2.addWeighted(store_bg, 0.7, heatmap_color, 0.3, 0)

    # Show live detection (for technical demo) and Dashboard (for business demo)
    cv2.imshow("Technical View: YOLO Detection", results[0].plot())
    cv2.imshow("Business Dashboard: Privacy Heatmap", dashboard_overlay)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 6. EXPORT DATA
print("Closing and saving Big Data...")
df = pd.DataFrame(analytics_log)
# Cleanse data: Remove low confidence detections (Threshold 0.6)
df_cleansed = df[df['AI_Confidence'] >= 0.6]
df_cleansed.to_csv("footfall_big_data_export.csv", index=False)
print(f"Exported {len(df_cleansed)} rows to CSV.")

cap.release()
cv2.destroyAllWindows()
