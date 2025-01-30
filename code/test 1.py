import cv2
import numpy as np
from tensorflow.keras.models import load_model
from mediapipe import solutions as mp

model = load_model('../models/dynamic_gesture_lstm.h5')
label_map = {0: 'hungry', 1: 'good morning', 2: 'thank you'}  # Update with your labels

mp_hands = mp.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_pose = mp.solutions.pose.Pose()

sequence = []
predictions = []
threshold = 0.8

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = mp_hands.process(rgb_frame)
    pose_results = mp_pose.process(rgb_frame)

    # Extract landmarks
    landmarks = []
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            landmarks.extend([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])

    if pose_results.pose_landmarks:
        landmarks.extend([(lm.x, lm.y, lm.z) for lm in pose_results.pose_landmarks.landmark])

    sequence.append(landmarks)
    sequence = sequence[-SEQUENCE_LENGTH:]

    if len(sequence) == SEQUENCE_LENGTH:
        # Make prediction
        res = model.predict(np.expand_dims(sequence, axis=0))[0]
        predictions.append(np.argmax(res))

        # Majority voting
        if len(predictions) > 5:
            predictions = predictions[-5:]

        current_pred = np.bincount(predictions).argmax()

        if res[current_pred] > threshold:
            # Display result
            cv2.putText(frame, f'{label_map[current_pred]} ({res[current_pred]:.2f})',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Sign Language Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()