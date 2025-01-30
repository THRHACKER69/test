import csv
import cv2
import os
import numpy as np
from hand_tracker import HandTracker

# Ensure data folder exists
data_folder = "../data"
csv_path = os.path.join(data_folder, "gestures_seq.csv")
os.makedirs(data_folder, exist_ok=True)

# Sequence parameters
max_seq_length = 20  # Ensure all sequences have 20 frames
num_features = 42  # Each frame has 21 hand landmarks (x, y)

# Create CSV file with headers if it doesn't exist
header = ["label", "sequence_id"] + [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)]
if not os.path.exists(csv_path):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

# Initialize Hand Tracker
tracker = HandTracker()
cap = cv2.VideoCapture(0)
label = input("Enter gesture label (e.g., Hello, Bye): ").strip()

sequence_id = 0  # Unique sequence number

print(f"📸 Collecting data for '{label}'... Press 'S' to start, 'ESC' to exit.")

while cap.isOpened():
    frames = []  # Store frames for one sequence
    frame_count = 0  # Count frames with hand detected

    for _ in range(max_seq_length):
        ret, frame = cap.read()
        if not ret:
            break

        frame, landmarks = tracker.track_hands(frame)
        cv2.imshow("Collect Dynamic Gesture Data", frame)

        # If landmarks detected, store them; otherwise, skip frame
        if landmarks and len(landmarks) == num_features:
            frames.append([label, sequence_id] + landmarks)
            frame_count += 1
        else:
            print("Skipping frame: No hand detected")

        key = cv2.waitKey(1)
        if key == 27:  # ESC to exit
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # Ensure sequence has exactly `max_seq_length` frames
    if frame_count < max_seq_length:
        print(f"Skipping sequence {sequence_id}: Less than {max_seq_length} frames with hand detected.")
    else:
        # Ensure sequence has exactly `max_seq_length` frames
        if len(frames) < max_seq_length:
            padding_frames = max_seq_length - len(frames)
            frames.extend([[label, sequence_id] + [0] * num_features] * padding_frames)

        # Save sequence to CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(frames)

        print(f"✅ Saved sequence {sequence_id} for '{label}'!")
        sequence_id += 1  # Increment sequence ID

cap.release()
cv2.destroyAllWindows()
