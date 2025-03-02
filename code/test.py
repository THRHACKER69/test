import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle

# Load the trained CNN model and label encoder
model = tf.keras.models.load_model('static_gesture_model.h5')
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[i].classification[0].label  # "Left" or "Right"

            # Extract landmarks (x, y coordinates)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])

            # Convert landmarks to NumPy array
            landmarks_np = np.array(landmarks).reshape(1, -1, 1)  # Reshape for CNN

            # Predict gesture using CNN model
            prediction = model.predict(landmarks_np)
            predicted_class = np.argmax(prediction)
            decoded_class = label_encoder.inverse_transform([predicted_class])[0]

            # Display prediction on the frame
            cv2.putText(frame, f"Gesture ({handedness}): {decoded_class}", (20, 50 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the video frame
    cv2.imshow("Static Gesture Recognition (CNN)", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
