import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque, Counter

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype() is deprecated")

# =========================
# CONFIG
# =========================
BASE_DIR = r"C:\Users\Leon\Desktop\OPER\models"
MODEL_PATH = BASE_DIR + r"\gesture_model.pkl"
LABEL_ENCODER_PATH = BASE_DIR + r"\label_encoder.pkl"

# =========================
# LOAD MODEL & LABELS
# =========================
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)
print("[INFO] Model and label encoder loaded")

# =========================
# MEDIAPIPE SETUP
# =========================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# =========================
# WEBCAM
# =========================
cap = cv2.VideoCapture(0)
print("[INFO] Starting real-time gesture detection (2 hands, dominant)...")
print("[INFO] Press 'Q' to quit")

# =========================
# SMOOTHING PARAMETERS
# =========================
history_len = 5
gesture_history = deque(maxlen=history_len)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    hand_labels = []

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # =========================
            # EXTRACT 9 FEATURES (wrist + index tip + middle tip, x,y,z)
            # =========================
            lm = hand_landmarks.landmark
            features = [
                lm[0].x, lm[0].y, lm[0].z,   # wrist
                lm[8].x, lm[8].y, lm[8].z,   # index tip
                lm[12].x, lm[12].y, lm[12].z # middle tip
            ]
            features = np.array(features).reshape(1, -1)

            # Predict gesture
            pred_idx = model.predict(features)[0]
            pred_label = label_encoder.inverse_transform([pred_idx])[0]
            hand_labels.append(pred_label)

    # =========================
    # DOMINANT LABEL WITH SMOOTHING
    # =========================
    if hand_labels:
        dominant_label = Counter(hand_labels).most_common(1)[0][0]
        gesture_history.append(dominant_label)
        smoothed_label = Counter(gesture_history).most_common(1)[0][0]
    else:
        smoothed_label = "No Hand"

    # =========================
    # DISPLAY RESULT
    # =========================
    cv2.putText(frame, f"Gesture: {smoothed_label}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("ASL Gesture Detection (Dominant)", frame)

    if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
