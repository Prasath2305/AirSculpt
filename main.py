import cv2
import mediapipe as mp
import pyautogui
import time
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

prev_x = None
cooldown = 0

def count_fingers(hand_landmarks):
    fingers = []

    # Thumb (skip for cursor control)
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Index, middle, ring, pinky
    for tip in [8, 12, 16, 20]:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)

while True:
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    h, w, _ = image.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            finger_count = count_fingers(hand_landmarks)
            cx = int(hand_landmarks.landmark[0].x * w)

            # === Cursor Move (1 finger) ===
            if finger_count == 1:
                if prev_x is not None and abs(cx - prev_x) > 5:
                    dx = cx - prev_x
                    pyautogui.moveRel(dx * 2, 0)
                    cooldown = 5

            # === Orbit View (4 fingers) ===
            elif finger_count == 4:
                if prev_x is not None and abs(cx - prev_x) > 10:
                    dx = cx - prev_x
                    pyautogui.mouseDown(button='middle')
                    pyautogui.moveRel(dx * 2, 0)
                    pyautogui.mouseUp(button='middle')
                    cooldown = 5

            # === Zoom In (5 fingers) ===
            elif finger_count == 5:
                pyautogui.scroll(30)
                cooldown = 10

            # === Zoom Out (fist / <=2 fingers) ===
            elif finger_count <= 2:
                pyautogui.scroll(-30)
                cooldown = 10

            prev_x = cx

    if cooldown > 0:
        cooldown -= 1

    cv2.imshow("AirSculpt", image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
