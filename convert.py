# =================================================
# VIDEO DATASET → CSV CONVERTER FOR 9 LANDMARK FEATURES
# =================================================

import os
import csv
import cv2
import mediapipe as mp

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
DATASET_DIR = "dataset_videos"         # folder with label subfolders
OUTPUT_CSV = "asl_hand_landmarks.csv"  # output CSV

FRAME_STRIDE = 2        # save every Nth frame
MIN_CONFIDENCE = 0.5
MAX_HANDS = 1            # only one hand per frame

# -------------------------------------------------
# MEDIAPIPE INITIALIZATION
# -------------------------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_HANDS,
    min_detection_confidence=MIN_CONFIDENCE,
    min_tracking_confidence=MIN_CONFIDENCE
)

# -------------------------------------------------
# UTILITY FUNCTIONS
# -------------------------------------------------
def validate_dataset(base_dir):
    if not os.path.isdir(base_dir):
        raise RuntimeError(f"Dataset folder not found: {base_dir}")
    labels = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    if not labels:
        raise RuntimeError("No label subfolders found in dataset_videos")
    return labels

# -------------------------------------------------
# CSV HEADER (9 features + label)
# -------------------------------------------------
CSV_HEADER = [
    "wrist_x", "wrist_y", "wrist_z",
    "index_x", "index_y", "index_z",
    "middle_x", "middle_y", "middle_z",
    "label"
]

# -------------------------------------------------
# MAIN PROCESS
# -------------------------------------------------
def main():
    labels = validate_dataset(DATASET_DIR)
    total_samples = 0

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)

        for label in labels:
            label_path = os.path.join(DATASET_DIR, label)

            for video_file in sorted(os.listdir(label_path)):
                if not video_file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    continue

                video_path = os.path.join(label_path, video_file)
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print("Skipping unreadable video:", video_path)
                    continue

                frame_idx = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_idx += 1
                    if frame_idx % FRAME_STRIDE != 0:
                        continue

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = hands.process(rgb)

                    if not result.multi_hand_landmarks:
                        continue

                    lm = result.multi_hand_landmarks[0].landmark

                    # Extract exactly wrist, index tip, middle tip
                    try:
                        row = [
                            round(lm[0].x, 6), round(lm[0].y, 6), round(lm[0].z, 6),   # wrist
                            round(lm[8].x, 6), round(lm[8].y, 6), round(lm[8].z, 6),   # index tip
                            round(lm[12].x, 6), round(lm[12].y, 6), round(lm[12].z, 6), # middle tip
                            label
                        ]
                    except Exception:
                        continue

                    writer.writerow(row)
                    total_samples += 1

                cap.release()

    hands.close()

    print("\n[SUCCESS]")
    print("Labels found   :", len(labels))
    print("Samples saved  :", total_samples)
    print("CSV output     :", OUTPUT_CSV)

# -------------------------------------------------
# ENTRY POINT
# -------------------------------------------------
if __name__ == "__main__":
    main()
