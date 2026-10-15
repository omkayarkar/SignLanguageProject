import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque, Counter

print("Starting live prediction... Press 'q' to quit.")

# Load trained model
model = joblib.load("sign_model.pkl")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

prediction_buffer = deque(maxlen=10)
CONFIDENCE_THRESHOLD = 0.80  # Internal threshold (not displayed)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    final_prediction = ""
    confidence = 0.0

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

            feature_vector = np.array(feature_vector).reshape(1, -1)

            probs = model.predict_proba(feature_vector)
            confidence = np.max(probs)
            prediction = model.classes_[np.argmax(probs)]

            if confidence > CONFIDENCE_THRESHOLD:
                prediction_buffer.append(prediction)
            else:
                prediction_buffer.clear()

    # Majority voting smoothing
    if prediction_buffer:
        final_prediction = Counter(prediction_buffer).most_common(1)[0][0]
    else:
        final_prediction = ""

    # Display only prediction
    cv2.putText(
        frame,
        f"Prediction: {final_prediction}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 0),
        3
    )

    cv2.imshow("Live Sign Language Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



"""
."""
