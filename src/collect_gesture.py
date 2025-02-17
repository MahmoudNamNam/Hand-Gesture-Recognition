import cv2
import mediapipe as mp
import csv

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Gesture labels
gesture_labels = {0: "No Key", 1: "Pause", 2: "Play", 3: "Like", 4: "Dislike"}

# Open CSV file to store landmarks
csv_file = open("../Data/gesture_data.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["label"] + [f"x{i},y{i},z{i}" for i in range(21)])  # 63 features total

cap = cv2.VideoCapture(0)

print("Press key 0-4 to label gestures. Press 'q' to quit.")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])  # Collecting 63 features (x, y, z)

            # Get user input for labeling
            key = cv2.waitKey(1) & 0xFF
            if key in [ord(str(i)) for i in range(5)]:
                label = int(chr(key))
                csv_writer.writerow([label] + landmarks)
                print(f"✅ Saved: {gesture_labels[label]} - {len(landmarks)} landmarks")

    cv2.imshow("Hand Gesture Collection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()
