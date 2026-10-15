import cv2
import mediapipe as mp
import numpy as np
import csv
import os

print("A–Z Dataset Collection Started")
print("Press A–Z to save sample")
print("Press Q to quit")

ALLOWED_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

file_name = "sign_dataset.csv"

# Create CSV file with header if it doesn't exist
if not os.path.exists(file_name):
    with open(file_name, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = [f"f{i+1}" for i in range(63)]
        header.append("label")
        writer.writerow(header)

sample_count = 0
last_saved_label = ""

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    feature_vector = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            feature_vector = []
            for lm in hand_landmarks.landmark:
                feature_vector.extend([lm.x, lm.y, lm.z])

    # Display info on screen
    cv2.putText(frame, f"Samples: {sample_count}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.putText(frame, f"Last Saved: {last_saved_label}",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 0, 0), 2)

    cv2.imshow("A-Z Dataset Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    # If pressed key is A-Z and hand detected
    if chr(key).upper() in ALLOWED_LABELS and feature_vector is not None:
        label = chr(key).upper()

        with open(file_name, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(feature_vector + [label])

        sample_count += 1
        last_saved_label = label

        print(f"Saved: {label} | Total Samples: {sample_count}")

cap.release()
cv2.destroyAllWindows()

"""
python create_dataset.py
python train_model.py
python predict_live.py

cd C:\Projects\SignLanguageProject
venv\Scripts\activate

"""