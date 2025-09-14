import cv2 # Import OpenCV library
import datetime as dt # Import Datetime library for timestamps

# Load OpenCV AI Classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam (select 1 for continuity camera macOS only)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert camera to greyscale
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        rect = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        time_now = dt.datetime.now().strftime('%d-%m-%yT%H:%M:%S')
        timestamp = cv2.putText(frame, str(time_now), (x, y+h+15), cv2.FONT_HERSHEY_PLAIN, 0.7, (250, 255, 0), 1)
        cv2.imwrite(time_now + '.jpg', frame) # Save Detection to file

    # Show new window with camera
    cv2.imshow("Attendance Tracker", frame)

    # Press Q to exit program.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()