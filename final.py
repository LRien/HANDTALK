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

warnings.filterwarnings("ignore")

# =========================
# CONFIG
# =========================
BASE_DIR = r"C:\Users\Leon\Desktop\OPER\models"
DATASET_DIR = r"C:\Users\Leon\Desktop\OPER\dataset"

MODEL_PATH = os.path.join(BASE_DIR, "gesture_model.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
LETTER_MODEL_PATH = os.path.join(BASE_DIR, "gesture_clf.pkl")
PALM_MODEL_PATH = os.path.join(BASE_DIR, "palm_detection_without_custom_op.tflite")
LANDMARK_MODEL_PATH = os.path.join(BASE_DIR, "hand_landmark.tflite")
ANCHORS_PATH = os.path.join(BASE_DIR, "anchors.csv")

CAM_W, CAM_H = 500, 560
HOLD_TIME = 0.5
GESTURE_BUFFER = 8
SMOOTH_FRAMES = 5
HAND_REMOVED_TIME = 1.0  # seconds user must remove hand

# =========================
# LOAD MODELS
# =========================
gesture_model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)
letter_model = joblib.load(LETTER_MODEL_PATH)
int_to_char = src.extra.classes

# =========================
# MEDIAPIPE
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
# HANDTRACKER NMS
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
target_letter = None
learning_feedback = ""
correct_count = 0
hand_removed = True
hand_removed_start = None

bbox_history = deque(maxlen=SMOOTH_FRAMES)
letter_buffer = deque(maxlen=GESTURE_BUFFER)
confirmed_letter = ""
pred_buffer = ""
gesture_start_time = 0
confidence_level = 0.0

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)

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

mode_label = tk.Label(root, text=f"MODE: {MODE}", fg="red",
                      bg="white", font=("Helvetica", 13, "bold"))
mode_label.pack()

# =========================
# BUTTONS
# =========================
btn_frame = tk.Frame(root, bg="white")
btn_frame.pack(pady=5)

def set_mode(m):
    global MODE, confirmed_letter, pred_buffer, gesture_start_time
    global correct_count, learning_feedback, hand_removed, hand_removed_start

    MODE = m
    confirmed_letter = ""
    pred_buffer = ""
    gesture_start_time = 0
    correct_count = 0
    learning_feedback = ""
    letter_buffer.clear()
    bbox_history.clear()
    hand_removed = True
    hand_removed_start = None
    update_reference_image()

ttk.Button(btn_frame, text="Gesture", command=lambda: set_mode("GESTURE")).pack(side="left", padx=6)
ttk.Button(btn_frame, text="Spelling", command=lambda: set_mode("SPELLING")).pack(side="left", padx=6)
ttk.Button(btn_frame, text="Learning", command=lambda: set_mode("LEARNING")).pack(side="left", padx=6)

# =========================
# TARGET LETTER
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
    global target_letter, correct_count, learning_feedback, hand_removed, hand_removed_start
    target_letter = letter_var.get()
    correct_count = 0
    learning_feedback = ""
    hand_removed = True
    hand_removed_start = None

letter_dropdown.bind("<<ComboboxSelected>>", update_target)

# =========================
# CAMERA DISPLAY
# =========================
camera_canvas = tk.Canvas(root, width=CAM_W, height=CAM_H, bg="black")
camera_canvas.pack(pady=10)
camera_label = tk.Label(camera_canvas, bg="black")
camera_label.place(relx=0.5, rely=0.5, anchor="center")

ref_canvas = tk.Canvas(root, width=150, height=150, bg="white")
ref_canvas.place(x=CAM_W+20, y=100)
ref_label = tk.Label(ref_canvas, bg="white")
ref_label.pack(expand=True)

def update_reference_image():
    if MODE == "LEARNING" and target_letter:
        folder = os.path.join(DATASET_DIR, target_letter)
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
detected_label = tk.Label(root, text="Detected: -", bg="white",
                          font=("Helvetica", 12, "bold"))
detected_label.pack()

confidence_label = tk.Label(root, text="Confidence: 0%", bg="white",
                            font=("Helvetica", 11, "bold"))
confidence_label.pack()

feedback_label = tk.Label(root, text="", bg="white", fg="green",
                          font=("Helvetica", 11, "bold"))
feedback_label.pack()

# =========================
# GUI UPDATE
# =========================
def update_gui(frame, detected, confidence):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (CAM_W, CAM_H))
    img = ImageTk.PhotoImage(Image.fromarray(frame))
    camera_label.imgtk = img
    camera_label.config(image=img)
    detected_label.config(text=f"Detected: {detected}")
    confidence_label.config(text=f"Confidence: {confidence*100:.1f}%")
    feedback_label.config(text=learning_feedback)
    mode_label.config(text=f"MODE: {MODE}")
    letter_dropdown.config(state="readonly" if MODE == "LEARNING" else "disabled")
    update_reference_image()

# =========================
# MAIN LOOP
# =========================
def loop():
    global confirmed_letter, pred_buffer, gesture_start_time
    global learning_feedback, correct_count, confidence_level
    global hand_removed, hand_removed_start

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        detected = "-"
        confidence = 0.0

        # SPELLING / LEARNING
        if MODE in ["SPELLING", "LEARNING"]:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            points, bboxes, joints = detector(rgb)

            # fix: evaluate bboxes safely
            hand_present = points is not None and joints is not None and bboxes is not None and len(bboxes) > 0

            # Hand removed logic for learning mode
            if MODE == "LEARNING" and hand_removed_start is not None:
                if hand_present:
                    hand_removed = False

            if not hand_present:
                if MODE == "LEARNING" and not hand_removed:
                    if hand_removed_start is None:
                        hand_removed_start = time.time()
                    elif time.time() - hand_removed_start >= HAND_REMOVED_TIME:
                        hand_removed = True
                        hand_removed_start = None
                        learning_feedback = f"Hand removed, please show sign again ({correct_count}/4)"

            if not hand_present:
                letter_buffer.clear()
                pred_buffer = ""
                confirmed_letter = ""
                gesture_start_time = 0
                detected = "-"
                confidence = 0.0
                update_gui(frame, detected, confidence)
                continue

            # Draw bbox
            box = bboxes[0] if len(bboxes) > 0 else None
            if box is not None and len(box) == 4:
                x1, y1, x2, y2 = map(int, box)
                bbox_history.append((x1, y1, x2, y2))

            if bbox_history:
                weights = np.linspace(1, 2, len(bbox_history))
                x1_avg = int(np.average([b[0] for b in bbox_history], weights=weights))
                y1_avg = int(np.average([b[1] for b in bbox_history], weights=weights))
                x2_avg = int(np.average([b[2] for b in bbox_history], weights=weights))
                y2_avg = int(np.average([b[3] for b in bbox_history], weights=weights))
                cv2.rectangle(frame, (x1_avg, y1_avg), (x2_avg, y2_avg), (0, 255, 0), 2)

            src.extra.draw_points(points, frame)
            pred = src.extra.predict_sign(joints, letter_model, int_to_char)
            letter_buffer.append(pred)

            most_common = Counter(letter_buffer).most_common(1)
            if most_common:
                stable, freq = most_common[0]
                confidence = freq / len(letter_buffer)
                if stable == pred_buffer:
                    if time.time() - gesture_start_time >= HOLD_TIME:
                        confirmed_letter = stable
                else:
                    pred_buffer = stable
                    gesture_start_time = time.time()

            detected = confirmed_letter

            # Learning logic: 4 correct detections, hand removed in between
            if MODE == "LEARNING" and target_letter and confirmed_letter and hand_removed:
                if confirmed_letter == target_letter:
                    correct_count = min(correct_count + 1, 4)
                    learning_feedback = f"Correct ({correct_count}/4) - Remove hand for next"
                    hand_removed = False
                    hand_removed_start = time.time()
                else:
                    learning_feedback = f"Wrong ({confirmed_letter})"

            cv2.putText(frame, f"LETTER: {confirmed_letter}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

        # GESTURE MODE
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
                    probas = gesture_model.predict_proba(features)[0]
                    confidence = max(probas)
                    labels.append(label_encoder.inverse_transform([pred])[0])

            detected = Counter(labels).most_common(1)[0][0] if labels else "No Hand"
            confidence = confidence if labels else 0.0

        update_gui(frame, detected, confidence)
        time.sleep(0.01)

# =========================
# START
# =========================
threading.Thread(target=loop, daemon=True).start()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
hands.close()
