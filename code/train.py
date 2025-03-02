import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load .npy files
X = np.load("../data/gestures_landmarks.npy",allow_pickle=True)  # Shape: (samples, features)
y = np.load("../data/gestures_labels.npy",allow_pickle=True)  # Shape: (samples,)

# Convert labels to numerical format
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split into training & testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape input for CNN (Adding channel dimension)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Define CNN Model
model = Sequential([
    Conv1D(64, kernel_size=3, activation="relu", input_shape=(X_train.shape[1], 1)),
    BatchNormalization(),
    Dropout(0.2),

    Conv1D(128, kernel_size=3, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(len(label_encoder.classes_), activation="softmax")  # Output layer
])

# Compile Model
model.compile(optimizer=Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train Model
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# Evaluate Accuracy
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Save the Model
model.save("static_gesture_model.h5")

# Save label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("Model saved as static_gesture_model.h5 and label_encoder.pkl")

# Load Model for Testing
model = keras.models.load_model("static_gesture_model.h5")
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Example Prediction
sample_input = X_test[0].reshape(1, X_test.shape[1], 1)
prediction = model.predict(sample_input)
predicted_class = np.argmax(prediction)
decoded_class = label_encoder.inverse_transform([predicted_class])

print(f"Predicted Gesture: {decoded_class[0]}")
