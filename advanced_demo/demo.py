import cv2
import numpy as np
import pandas as pd
import time
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# SETUP: Load the "Still Image" of the empty store for the dashboard
background_store_img = cv2.imread("store_layout.jpg") 
cap = cv2.VideoCapture(0)

# Tracker for Dwell Time and Engagement
track_history = {} # {id: {'start': time, 'hits': 0, 'last_coord': (x,y)}}

def get_zone(x, y, w, h):
    # Logic to categorize coordinates into store zones
    if x < w/3: return "Entrance"
    elif x > 2*w/3: return "Checkout"
    else: return "Aisle_1"

# Initialize Data Collection List
tabular_data = []

while True:
    ret, frame = cap.read()
    if not ret: break
    h, w, _ = frame.shape

    results = model.track(frame, persist=True, verbose=False) # Use track for ID persistence
    
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()

        for box, track_id, conf in zip(boxes, track_ids, confidences):
            cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
            
            # 1. Logic for Dwell Time and Engagement
            if track_id not in track_history:
                track_history[track_id] = {'start': time.time(), 'hits': 1}
            else:
                track_history[track_id]['hits'] += 1 # Every frame detected counts toward engagement intensity
            
            # 2. Log Data for Tabular Output (Unit 10 Task 4)
            current_dwell = time.time() - track_history[track_id]['start']
            
            # We calculate "Engagement" based on the formula we discussed
            engagement_score = (current_dwell * 0.5) + (track_history[track_id]['hits'] * 0.2)

            tabular_data.append({
                "DetectionID": track_id,
                "Timestamp": pd.Timestamp.now(),
                "Zone": get_zone(cx, cy, w, h),
                "X_Coord": cx,
                "Y_Coord": cy,
                "DwellTime_s": round(current_dwell, 2),
                "EngagementScore": round(engagement_score, 2),
                "AI_Confidence": round(float(conf), 2)
            })

    # Dashoard logic: Overlaying heatmap on the STILL IMAGE instead of live video
    # (Heatmap generation code remains similar to your original blob logic)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# Save the captured big data to CSV for Task 4 analysis
df = pd.DataFrame(tabular_data)
df.to_csv("ai_analytics_output.csv", index=False)

cap.release()
cv2.destroyAllWindows()
