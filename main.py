import cv2
import time
from src.hand_tracker import HandTracker


def main():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()

    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Could not read frame.")
            break

        frame = cv2.flip(frame, 1)
        timestamp_ms = int(time.time() * 1000)

        frame = tracker.find_hands(frame, timestamp_ms)
        landmarks = tracker.find_position(frame)

        if landmarks:
            hand_label = tracker.get_hand_label()
            fingers = tracker.fingers_up(landmarks, hand_label)
            up_finger_names = tracker.get_up_finger_names(fingers)

            if up_finger_names:
                finger_text = ", ".join(up_finger_names)
            else:
                finger_text = "No fingers up"

            cv2.putText(
                frame,
                f"Hand: {hand_label}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Up Fingers: {finger_text}",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Finger State: {fingers}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 200, 100),
                2
            )

        cv2.putText(
            frame,
            "AirAuth-SOC | Press Q to quit",
            (10, 470),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow("AirAuth-SOC", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
