import cv2
import mediapipe as mp
import joblib
import numpy as np


# Load the trained SVM model (same as your original approach)
svm_model = joblib.load('../models/static_gesture_svm.pkl')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmarks (x, y coordinates)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])

            # Predict gesture using the SVM model (same as your original code)
            svm_prediction = svm_model.predict([landmarks])  # Pass raw landmarks directly
            predicted_label = svm_prediction[0]  # Predicted gesture label (e.g., "A")

            # Display prediction on the frame
            cv2.putText(frame, f"Gesture: {predicted_label}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the video frame
    cv2.imshow("Static Gesture Recognition with SVM", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


