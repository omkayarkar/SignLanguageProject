import cv2
import mediapipe as mp
import numpy as np

print("Starting hand tracking... Press 'q' to quit.")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Detect both hands
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Open default camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame.")
        break

    # Mirror view
    frame = cv2.flip(frame, 1)

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks and results.multi_handedness:

        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

            # Get left or right hand label
            hand_label = results.multi_handedness[idx].classification[0].label

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Extract feature vector (63 values per hand)
            feature_vector = []

            for lm in hand_landmarks.landmark:
                feature_vector.extend([lm.x, lm.y, lm.z])

            feature_vector = np.array(feature_vector)

            print(f"{hand_label} Hand Feature Vector Length:", len(feature_vector))

            # Display hand label on screen
            cv2.putText(
                frame,
                hand_label,
                (10, 30 + idx * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    cv2.imshow("Hand Tracking - Both Hands", frame)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

"""python collect_data.py"""