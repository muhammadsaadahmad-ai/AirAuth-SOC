import cv2
import time
from src.hand_tracker import HandTracker


def main():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()

    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    canvas = None
    prev_x, prev_y = 0, 0
    draw_color = (0, 0, 255)
    brush_thickness = 5

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Could not read frame.")
            break

        frame = cv2.flip(frame, 1)

        if canvas is None:
            canvas = frame.copy() * 0

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

            # Index fingertip position
            x, y = landmarks[8][1], landmarks[8][2]

            # DRAW MODE: only index finger up
            if fingers == [0, 1, 0, 0, 0]:
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y

                cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, brush_thickness)
                prev_x, prev_y = x, y
            else:
                prev_x, prev_y = 0, 0

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

            if fingers == [0, 1, 0, 0, 0]:
                cv2.putText(
                    frame,
                    "Mode: DRAW",
                    (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )
            else:
                cv2.putText(
                    frame,
                    "Mode: IDLE",
                    (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (200, 200, 200),
                    2
                )

        else:
            prev_x, prev_y = 0, 0

        gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        frame = cv2.bitwise_and(frame, mask)
        frame = cv2.bitwise_or(frame, canvas)

        cv2.putText(
            frame,
            "AirAuth-SOC | Index only = Draw | Press Q to quit",
            (10, 470),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
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
