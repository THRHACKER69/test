import csv
import cv2
import os
from hand_tracker import HandTracker

# Ensure data folder exists
data_folder = "../data"
csv_path = os.path.join(data_folder, "gestures.csv")
os.makedirs(data_folder, exist_ok=True)

# Create CSV file with headers if it doesn't exist
header = ['label'] + [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)]
if not os.path.exists(csv_path):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

# Initialize Hand Tracker
tracker = HandTracker()
cap = cv2.VideoCapture(0)
label = "Z"  # Change this for different gestures

# Open CSV once to avoid frequent writes
with open(csv_path, 'a', newline='') as f:
    writer = csv.writer(f)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, landmarks = tracker.track_hands(frame)
        cv2.imshow("Collect Data", frame)

        key = cv2.waitKey(1)
        if key == ord('s') and landmarks:
            if len(landmarks) == 42:  # Ensure correct landmark count
                writer.writerow([label] + landmarks)
                print(f"Saved sample for {label}!")
            else:
                print(f"Skipping frame, incorrect landmark count: {len(landmarks)}")

        elif key == 27:
            break

cap.release()
cv2.destroyAllWindows()
