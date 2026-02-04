# hand_letter_worker.py
import cv2
from src.hand_tracker_nms import HandTrackerNMS
import src.extra
import joblib
import warnings
import numpy as np
import time
from collections import deque, Counter

def run_letter_detector(on_letter, stop_flag, camera_index=0):
    """
    Functionalized letter detection script for SPELLING mode.

    Args:
        on_letter: function(letter) -> callback when a letter is confirmed
        stop_flag: function() -> returns True when the loop should stop
        camera_index: which camera to use (default 0)
    """

    # =========================
    # SUPPRESS WARNINGS
    # =========================
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    PALM_MODEL_PATH = "models/palm_detection_without_custom_op.tflite"
    LANDMARK_MODEL_PATH = "models/hand_landmark.tflite"
    ANCHORS_PATH = "models/anchors.csv"

    int_to_char = src.extra.classes
    connections = src.extra.connections

    # =========================
    # INIT DETECTOR AND MODEL
    # =========================
    detector = HandTrackerNMS(
        PALM_MODEL_PATH,
        LANDMARK_MODEL_PATH,
        ANCHORS_PATH,
        box_shift=0.2,
        box_enlarge=1.3
    )

    gesture_clf = joblib.load("models/gesture_clf.pkl")

    # =========================
    # CAMERA
    # =========================
    capture = cv2.VideoCapture(camera_index)
    flip_camera = False

    letter = ""
    pred_buffer = ""
    gesture_start_time = 0
    HOLD_TIME = 0.5

    # Smoothing buffers
    SMOOTH_FRAMES = 5
    bbox_history = deque(maxlen=SMOOTH_FRAMES)

    GESTURE_BUFFER = 8
    gesture_history = deque(maxlen=GESTURE_BUFFER)

    # =========================
    # MAIN LOOP
    # =========================
    while capture.isOpened():

        # Stop if UI requests
        if stop_flag():
            break

        hasFrame, frame = capture.read()
        if not hasFrame:
            break

        if flip_camera:
            frame = cv2.flip(frame, 1)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        points, bboxes, joints = detector(image)

        if points is not None and bboxes is not None and len(bboxes) > 0:
            box = bboxes[0]
            if len(box) == 4:
                x1, y1, x2, y2 = map(int, box)
                bbox_history.append((x1, y1, x2, y2))

            # Smooth bounding box (weighted average)
            if bbox_history:
                weights = np.linspace(1, 2, len(bbox_history))
                x1_avg = int(np.average([b[0] for b in bbox_history], weights=weights))
                y1_avg = int(np.average([b[1] for b in bbox_history], weights=weights))
                x2_avg = int(np.average([b[2] for b in bbox_history], weights=weights))
                y2_avg = int(np.average([b[3] for b in bbox_history], weights=weights))
                # Optional: you can remove this cv2.rectangle if you don't want visualization
                # cv2.rectangle(frame, (x1_avg, y1_avg), (x2_avg, y2_avg), (0, 255, 0), 2)

            # Draw landmarks (optional, can be skipped in SPELLING)
            # src.extra.draw_points(points, frame)

            # Predict gesture / letter
            pred_sign = src.extra.predict_sign(joints, gesture_clf, int_to_char)
            gesture_history.append(pred_sign)

            # Majority vote
            most_common = Counter(gesture_history).most_common(1)
            if most_common:
                stable_sign = most_common[0][0]

                # Confirm if held for HOLD_TIME
                if stable_sign == pred_buffer:
                    if time.time() - gesture_start_time >= HOLD_TIME:
                        if stable_sign != letter:
                            letter = stable_sign
                            on_letter(letter)  # send confirmed letter to UI
                else:
                    pred_buffer = stable_sign
                    gesture_start_time = time.time()

        # Optional sleep to reduce CPU
        time.sleep(0.005)

    # Release camera when done
    capture.release()
