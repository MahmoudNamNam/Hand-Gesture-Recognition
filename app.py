import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib

# Load trained model and scaler
try:
    model = tf.keras.models.load_model("./models/gesture_model.h5")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()
try:
    scaler = joblib.load("./models/scaler.pkl")
    print("✅ Scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading scaler: {e}")
    exit()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Define gesture labels
gesture_labels = {0: "No Key", 1: "Pause", 2: "Play", 3: "Like", 4: "Dislike"}

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

# Store last detected gesture
last_gesture = "No Key"

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret or frame is None:
        print("❌ Error: Failed to capture frame.")
        break

    # Flip and convert frame for MediaPipe processing
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand detection
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            try:
                # Extract hand landmarks
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                # Ensure correct number of landmarks before passing to model
                expected_landmark_count = model.input_shape[1]  # Typically 63 (21 landmarks * 3)
                if len(landmarks) != expected_landmark_count:
                    print(f"⚠ Unexpected number of landmarks: {len(landmarks)}")
                    continue

                # Normalize and predict
                landmarks = np.array(landmarks).reshape(1, -1)
                landmarks = scaler.transform(landmarks)
                prediction = model.predict(landmarks)
                predicted_class = np.argmax(prediction)
                last_gesture = gesture_labels.get(predicted_class, "Unknown")

            except Exception as e:
                print(f"❌ Error processing hand landmarks: {e}")
                continue
    else:
        last_gesture = "No Hand Detected"

    # Display the label in the center-top of the screen
    cv2.putText(frame, f"Gesture: {last_gesture}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)

    # Show the frame
    cv2.imshow("Gesture Recognition", frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
