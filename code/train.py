import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Load Static Gesture Data
print("\nLoading static gesture dataset...")
static_data = pd.read_csv("../data/gestures.csv")
X_static = static_data.drop("label", axis=1)
y_static = static_data["label"]

# Train Static Gesture Model (SVM)
X_train_stat, X_test_stat, y_train_stat, y_test_stat = train_test_split(
    X_static, y_static, test_size=0.2, random_state=42
)
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_stat, y_train_stat)
y_pred_stat = svm_model.predict(X_test_stat)
accuracy_stat = accuracy_score(y_test_stat, y_pred_stat)
print(f"\n✅ Static Gesture Model Accuracy: {accuracy_stat * 100:.2f}%")
joblib.dump(svm_model, "../models/static_gesture_svm.pkl")
print("📁 SVM model saved as '../models/static_gesture_svm.pkl'.")