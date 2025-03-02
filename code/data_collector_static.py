import cv2
import os
import numpy as np
from hand_tracker import HandTracker

# Ensure data folder exists
data_folder = "data"
npy_landmarks_path = os.path.join(data_folder, "gestures_landmarks.npy")
npy_labels_path = os.path.join(data_folder, "gestures_labels.npy")
os.makedirs(data_folder, exist_ok=True)

# Initialize Hand Tracker
tracker = HandTracker()
cap = cv2.VideoCapture(0)
label = input("Enter name of the gesture:")  # Change this for different gestures

# Load existing data if available
if os.path.exists(npy_landmarks_path) and os.path.exists(npy_labels_path):
    existing_landmarks = np.load(npy_landmarks_path).tolist()
    existing_labels = np.load(npy_labels_path).tolist()
else:
    existing_landmarks = []
    existing_labels = []

print(f"📂 Existing data loaded: {len(existing_labels)} samples")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame, landmarks = tracker.track_hands(frame)
    cv2.imshow("Collect Data", frame)

    key = cv2.waitKey(1)
    if key == ord('s') and landmarks:
        if len(landmarks) == 42:  # Ensure correct landmark count
            existing_landmarks.append(landmarks)
            existing_labels.append(label)
            print(f"✅ Saved sample for {label}")
        else:
            print(f"⚠ Skipping frame, incorrect landmark count: {len(landmarks)}")

    elif key == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()

# Convert lists to NumPy arrays and save
np.save(npy_landmarks_path, np.array(existing_landmarks, dtype=np.float32))
np.save(npy_labels_path, np.array(existing_labels, dtype=object))  # Keep as object type for labels

print(f"📁 Saved landmarks: {npy_landmarks_path} ({len(existing_landmarks)} samples)")
print(f"📁 Saved labels: {npy_labels_path} ({len(existing_labels)} samples)")
