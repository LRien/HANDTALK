# train_gesture_model_9distances.py
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import joblib

# -------------------------------
# LOAD DATA
# -------------------------------
# CSV should have 9 distance columns + 'label' column
data = pd.read_csv(r"C:\Users\Leon\Desktop\OPER\data\alphabet.csv")  # update path

# Features (X) and labels (y)
X = data.drop("label", axis=1).values  # 9 distance features
y = data["label"].values               # labels

print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"Unique labels: {np.unique(y)}")

# -------------------------------
# SPLIT TRAIN/TEST
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=9
)

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

# -------------------------------
# TRAIN GAUSSIAN NB
# -------------------------------
clf = GaussianNB()
clf.fit(X_train, y_train)

# Evaluate
score = clf.score(X_test, y_test)
print(f"Model Accuracy: {score*100:.2f}%")

# -------------------------------
# SAVE MODEL
# -------------------------------
joblib.dump(clf, r"C:\Users\Leon\Desktop\OPER\models\gesture_clf.pkl", compress=True)
print("Model saved to models\\gesture_clf.pkl")
