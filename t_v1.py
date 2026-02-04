import os
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# CONFIG
# =========================
BASE_DIR = r"C:\Users\Leon\Desktop\OPER"
DATA_PATH = os.path.join(BASE_DIR, "data", "asl_hand_landmarks.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "gesture_model.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)
print(f"[INFO] Dataset loaded: {df.shape}")

# =========================
# SELECT ONLY THE 9 FEATURES
# =========================
FEATURE_COLUMNS = [
    'wrist_x', 'wrist_y', 'wrist_z',
    'index_x', 'index_y', 'index_z',
    'middle_x', 'middle_y', 'middle_z'
]

for col in FEATURE_COLUMNS:
    if col not in df.columns:
        raise ValueError(f"Missing feature column in CSV: {col}")

X = df[FEATURE_COLUMNS].astype(np.float32)
y = df["label"]

# =========================
# LABEL ENCODING
# =========================
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("[INFO] Gesture Classes:")
for i, cls in enumerate(label_encoder.classes_):
    print(f"  {i} -> {cls}")

# =========================
# VISUALIZE CLASS DISTRIBUTION
# =========================
plt.figure(figsize=(14, 6))
sns.countplot(x=y)
plt.title("Gesture Class Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.25,
    random_state=42,
    stratify=y_encoded
)

print(f"[INFO] Train samples: {X_train.shape[0]}")
print(f"[INFO] Test samples : {X_test.shape[0]}")

# =========================
# TRAIN MODEL
# =========================
model = GaussianNB()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"[RESULT] Accuracy: {accuracy * 100:.2f}%")

# =========================
# SAVE MODEL & LABEL ENCODER
# =========================
joblib.dump(model, MODEL_PATH)
joblib.dump(label_encoder, LABEL_ENCODER_PATH)

print(f"[SAVED] Model → {MODEL_PATH}")
print(f"[SAVED] Label Encoder → {LABEL_ENCODER_PATH}")

# =========================
# SAMPLE PREDICTION CHECK
# =========================
sample_idx = 0
sample = X_test.iloc[sample_idx].values.reshape(1, -1)

pred_idx = model.predict(sample)[0]
pred_label = label_encoder.inverse_transform([pred_idx])[0]
true_label = label_encoder.inverse_transform([y_test[sample_idx]])[0]

print(f"[TEST] Predicted: {pred_label} | Actual: {true_label}")
