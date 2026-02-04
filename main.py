import cv2
import mediapipe as mp
import numpy as np
import joblib
import threading
import time
import os
import warnings
import tkinter as tk
from tkinter import ttk, font
from PIL import Image, ImageTk
from collections import deque, Counter

from src.hand_tracker_nms import HandTrackerNMS
import src.extra

# =========================
# SUPPRESS WARNINGS
# =========================
warnings.filterwarnings("ignore")

# =========================
# CONFIG
# =========================
BASE_DIR = r"C:\Users\Leon\Desktop\OPER\models"
DATASET_DIR = r"C:\Users\Leon\Desktop\OPER\dataset"
LETTER_IMG_DIR = DATASET_DIR

MODEL_PATH = BASE_DIR + r"\gesture_model.pkl"
LABEL_ENCODER_PATH = BASE_DIR + r"\label_encoder.pkl"
LETTER_MODEL_PATH = BASE_DIR + r"\gesture_clf.pkl"
PALM_MODEL_PATH = BASE_DIR + r"\palm_detection_without_custom_op.tflite"
LANDMARK_MODEL_PATH = BASE_DIR + r"\hand_landmark.tflite"
ANCHORS_PATH = BASE_DIR + r"\anchors.csv"

MIN_CONFIDENCE = 0.55
VOTE_WINDOW = 8
LETTER_HOLD_TIME = 1.0
LETTER_COOLDOWN = 0.3

# =========================
# LOAD MODELS
# =========================
gesture_model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)
letter_model = joblib.load(LETTER_MODEL_PATH)
int_to_char = src.extra.classes

# =========================
# MEDIAPIPE (GESTURE)
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
# HANDTRACKER NMS (LETTERS)
# =========================
detector = HandTrackerNMS(
    PALM_MODEL_PATH,
    LANDMARK_MODEL_PATH,
    ANCHORS_PATH,
    box_shift=0.2,
    box_enlarge=1.3
)

# =========================
# STATE
# =========================
MODE = "GESTURE"
confirmed_word = ""

current_letter = None
letter_hold_start = 0
last_letter_time = 0
letter_votes = deque(maxlen=VOTE_WINDOW)

target_letter = None
learning_feedback = ""
correct_count = 0

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)
CAM_W, CAM_H = 500, 560

# =========================
# FEATURE NORMALIZATION
# =========================
def normalize_joints_for_gnb(joints):
    def safe(lm):
        return [lm[i] if len(lm) > i else 0.0 for i in range(3)]

    lm0 = safe(joints[0] if len(joints) > 0 else [])
    lm8 = safe(joints[8] if len(joints) > 8 else [])
    lm12 = safe(joints[12] if len(joints) > 12 else [])
    return np.array(lm0 + lm8 + lm12).reshape(1, -1)

# =========================
# HAND ORIENTATION FILTER
# =========================
def is_hand_facing_camera(joints):
    if len(joints) < 18:
        return False

    index_mcp = np.array(joints[5][:2])
    pinky_mcp = np.array(joints[17][:2])
    palm_width = np.linalg.norm(index_mcp - pinky_mcp)
    palm_height = abs(joints[0][1] - joints[9][1])

    return palm_width > palm_height * 0.6

# =========================
# GUI SETUP
# =========================
root = tk.Tk()
root.title("HandTalk")
root.geometry(f"{CAM_W+200}x900")
root.configure(bg="white")
root.resizable(False, False)

title_font = font.Font(size=18, weight="bold")
tk.Label(root, text="HANDTALK", bg="white", font=title_font).pack(pady=10)

mode_label = tk.Label(root, text="MODE: GESTURE", fg="red",
                      bg="white", font=("Helvetica", 13, "bold"))
mode_label.pack()

btn_frame = tk.Frame(root, bg="white")
btn_frame.pack(pady=5)

def set_mode(m):
    global MODE, confirmed_word, correct_count, learning_feedback
    MODE = m
    confirmed_word = ""
    correct_count = 0
    learning_feedback = ""
    letter_votes.clear()
    update_reference_image()

ttk.Button(btn_frame, text="Gesture", command=lambda: set_mode("GESTURE")).pack(side="left", padx=6)
ttk.Button(btn_frame, text="Spelling", command=lambda: set_mode("SPELLING")).pack(side="left", padx=6)
ttk.Button(btn_frame, text="Learning", command=lambda: set_mode("LEARNING")).pack(side="left", padx=6)

# =========================
# LEARNING DROPDOWN
# =========================
letter_frame = tk.Frame(root, bg="white")
letter_frame.pack(pady=5)

tk.Label(letter_frame, text="Target Letter:", bg="white",
         font=("Helvetica", 11, "bold")).pack(side="left", padx=5)

letter_var = tk.StringVar(value="A")
letter_dropdown = ttk.Combobox(
    letter_frame,
    textvariable=letter_var,
    values=[chr(i) for i in range(65, 91)],
    state="readonly",
    width=5
)
letter_dropdown.pack(side="left")

def update_target(event=None):
    global target_letter, correct_count, learning_feedback
    target_letter = letter_var.get()
    correct_count = 0
    learning_feedback = ""
    update_reference_image()

letter_dropdown.bind("<<ComboboxSelected>>", update_target)

# =========================
# CAMERA CANVAS
# =========================
camera_canvas = tk.Canvas(root, width=CAM_W, height=CAM_H, bg="black")
camera_canvas.pack(pady=10)

camera_label = tk.Label(camera_canvas, bg="black")
camera_label.place(relx=0.5, rely=0.5, anchor="center")

# =========================
# REFERENCE IMAGE
# =========================
ref_canvas = tk.Canvas(root, width=150, height=150, bg="white")
ref_canvas.place(x=CAM_W+20, y=100)

ref_label = tk.Label(ref_canvas, bg="white")
ref_label.pack(expand=True)

def update_reference_image():
    if MODE == "LEARNING" and target_letter:
        folder = os.path.join(LETTER_IMG_DIR, target_letter)
        if os.path.isdir(folder):
            for f in os.listdir(folder):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img = Image.open(os.path.join(folder, f)).resize((150, 150))
                    imgtk = ImageTk.PhotoImage(img)
                    ref_label.imgtk = imgtk
                    ref_label.config(image=imgtk)
                    return
    ref_label.config(image="", text="")

# =========================
# INFO LABELS
# =========================
detected_label = tk.Label(root, text="Detected: -", bg="white", font=("Helvetica", 12, "bold"))
detected_label.pack()

target_label = tk.Label(root, text="Target: -", bg="white", fg="blue",
                        font=("Helvetica", 12, "bold"))
target_label.pack()

progress = ttk.Progressbar(root, length=300, maximum=3)
progress.pack(pady=6)

feedback_label = tk.Label(root, text="", bg="white", fg="green",
                          font=("Helvetica", 11, "bold"))
feedback_label.pack()

# =========================
# BACKSPACE SUPPORT
# =========================
def on_backspace(event):
    global confirmed_word
    if MODE == "SPELLING" and confirmed_word:
        confirmed_word = confirmed_word[:-1]

root.bind("<BackSpace>", on_backspace)

# =========================
# GUI UPDATE
# =========================
def update_gui(frame, detected):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (CAM_W, CAM_H))
    img = ImageTk.PhotoImage(Image.fromarray(frame))
    camera_label.imgtk = img
    camera_label.config(image=img)

    detected_label.config(text=f"Detected: {detected if MODE!='SPELLING' else '—'}")
    target_label.config(text=f"Target: {target_letter or '-'}")
    feedback_label.config(text=learning_feedback)
    progress["value"] = correct_count
    mode_label.config(text=f"MODE: {MODE}")

    letter_dropdown.config(state="readonly" if MODE == "LEARNING" else "disabled")
    update_reference_image()

# =========================
# MAIN LOOP
# =========================
def loop():
    global current_letter, last_letter_time, confirmed_word
    global letter_hold_start, correct_count, learning_feedback

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        now = time.time()
        detected = "-"

        if MODE in ["SPELLING", "LEARNING"]:
            points, _, joints = detector(frame)

            if points is not None and joints is not None:
                if not is_hand_facing_camera(joints):
                    update_gui(frame, detected)
                    continue

                src.extra.draw_points(points, frame)
                features = normalize_joints_for_gnb(joints)
                probs = letter_model.predict_proba(features)[0]

                idx = np.argmax(probs)
                conf = probs[idx]

                if conf >= MIN_CONFIDENCE:
                    letter_votes.append(int_to_char[idx])

                if len(letter_votes) == VOTE_WINDOW:
                    stable = Counter(letter_votes).most_common(1)[0][0]

                    thumb_tip = joints[4]
                    index_mcp = joints[5]
                    if stable in ["L", "P"]:
                        stable = "L" if thumb_tip[0] < index_mcp[0] else "P"

                    detected = stable

                    if MODE == "LEARNING" and target_letter:
                        if stable == target_letter:
                            correct_count = min(correct_count + 1, 3)
                            learning_feedback = f"Correct ({correct_count}/3)"
                        else:
                            learning_feedback = f"Wrong ({stable})"

                    elif MODE == "SPELLING":
                        if stable != current_letter:
                            current_letter = stable
                            letter_hold_start = now
                        elif now - letter_hold_start >= LETTER_HOLD_TIME:
                            if now - last_letter_time >= LETTER_COOLDOWN:
                                confirmed_word += current_letter
                                last_letter_time = now
                                current_letter = None
                                letter_votes.clear()

                cv2.putText(frame, f"WORD: {confirmed_word}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            labels = []

            if result.multi_hand_landmarks:
                for h in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS)
                    lm = h.landmark
                    features = np.array([
                        lm[0].x, lm[0].y, lm[0].z,
                        lm[8].x, lm[8].y, lm[8].z,
                        lm[12].x, lm[12].y, lm[12].z
                    ]).reshape(1, -1)

                    pred = gesture_model.predict(features)[0]
                    labels.append(label_encoder.inverse_transform([pred])[0])

            detected = Counter(labels).most_common(1)[0][0] if labels else "No Hand"

        update_gui(frame, detected)
        time.sleep(0.01)

# =========================
# START
# =========================
threading.Thread(target=loop, daemon=True).start()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
hands.close()
