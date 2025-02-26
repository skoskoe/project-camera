import cv2
from ultralytics import YOLO

# Load the model from a local file instead of Hugging Face
model = YOLO("yolov8n-face.pt")  # File should be in the same folder
  # Ensure the file is in the same directory

# Open the laptop camera
cap = cv2.VideoCapture(0)

frame_count = 0  # Counter to track frames

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1  # Increase frame count

    if frame_count % 10 == 0:  # Process only every 10th frame
        results = model(frame)

        # Draw bounding boxes on detected faces
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("YOLOv8 Face Detection - Every 10th Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
