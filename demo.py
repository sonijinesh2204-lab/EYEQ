import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.75, min_tracking_confidence=0.75)
mp_draw = mp.solutions.drawing_utils

def is_sos(landmarks):
    tips = [8, 12, 16, 20]
    knuckles = [6, 10, 14, 18]
    folded = all(landmarks[t].y > landmarks[k].y for t, k in zip(tips, knuckles))
    thumb = abs(landmarks[4].x - landmarks[2].x) < 0.12
    return folded and thumb

cap = cv2.VideoCapture(0)
print("Camera running. Make a closed fist for SOS. Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    sos_detected = False

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            if is_sos(hand.landmark):
                sos_detected = True

    if sos_detected:
        cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 255), -1)
        cv2.putText(frame, "!! SOS ALERT - HELP NEEDED !!", (10, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
        print("ALERT: SOS Gesture Detected!")
    else:
        cv2.putText(frame, "Monitoring... Show closed fist for SOS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("EYEQ - SOS Detection", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()