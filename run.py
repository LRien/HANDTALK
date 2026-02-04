import cv2
from src.hand_tracker_nms import HandTrackerNMS
import src.extra
import joblib
import warnings
import numpy as np
import time
from collections import deque, Counter

# =========================
# SUPPRESS WARNINGS
# =========================
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

WINDOW = "Hand Tracking"
PALM_MODEL_PATH = "models/palm_detection_without_custom_op.tflite"
LANDMARK_MODEL_PATH = "models/hand_landmark.tflite"
ANCHORS_PATH = "models/anchors.csv"

connections = src.extra.connections
int_to_char = src.extra.classes

detector = HandTrackerNMS(
    PALM_MODEL_PATH,
    LANDMARK_MODEL_PATH,
    ANCHORS_PATH,
    box_shift=0.2,
    box_enlarge=1.3
)

gesture_clf = joblib.load(r'models\\gesture_clf.pkl')

cv2.namedWindow(WINDOW)
capture = cv2.VideoCapture(0)

flip_camera = False
letter = ""
pred_buffer = ""
gesture_start_time = 0
HOLD_TIME = 0.5  # seconds required to hold gesture

# Bounding box smoothing
SMOOTH_FRAMES = 5
bbox_history = deque(maxlen=SMOOTH_FRAMES)

# Gesture prediction smoothing
GESTURE_BUFFER = 8  # number of frames to consider
gesture_history = deque(maxlen=GESTURE_BUFFER)

while True:
    hasFrame, frame = capture.read()
    if not hasFrame:
        break

    if flip_camera:
        frame = cv2.flip(frame, 1)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    points, bboxes, joints = detector(image)

    pred_sign = ""

    if points is not None and bboxes is not None and len(bboxes) > 0:
        # Only take the first detected hand
        box = bboxes[0]
        if len(box) == 4:
            x1, y1, x2, y2 = map(int, box)
            bbox_history.append((x1, y1, x2, y2))

        # Smooth bounding box (weighted average)
        if len(bbox_history) > 0:
            weights = np.linspace(1, 2, len(bbox_history))  # more weight to recent frames
            x1_avg = int(np.average([b[0] for b in bbox_history], weights=weights))
            y1_avg = int(np.average([b[1] for b in bbox_history], weights=weights))
            x2_avg = int(np.average([b[2] for b in bbox_history], weights=weights))
            y2_avg = int(np.average([b[3] for b in bbox_history], weights=weights))
            cv2.rectangle(frame, (x1_avg, y1_avg), (x2_avg, y2_avg), (0, 255, 0), 2)

        # Draw landmarks
        src.extra.draw_points(points, frame)

        # Predict gesture
        pred_sign = src.extra.predict_sign(joints, gesture_clf, int_to_char)
        gesture_history.append(pred_sign)

        # Use majority vote over last GESTURE_BUFFER frames
        most_common = Counter(gesture_history).most_common(1)
        if most_common:
            stable_sign = most_common[0][0]

            # Only register if held for HOLD_TIME
            if stable_sign == pred_buffer:
                if time.time() - gesture_start_time >= HOLD_TIME:
                    letter = stable_sign
            else:
                pred_buffer = stable_sign
                gesture_start_time = time.time()

    # Display only the current label
    if letter != "":
        # Highlight label for better visibility
        cv2.rectangle(frame, (40, 20), (220, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"Letter: {letter}", (50, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    cv2.imshow(WINDOW, frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break
    elif key == 8:  # BACKSPACE
        letter = ""
        pred_buffer = ""
        gesture_start_time = 0
        bbox_history.clear()
        gesture_history.clear()
    elif key == ord('f') or key == ord('F'):  # Flip camera
        flip_camera = not flip_camera

capture.release()
cv2.destroyAllWindows()
