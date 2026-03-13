import cv2
import time
from src.hand_tracker import HandTracker


def main():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()

    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    canvas = None
    prev_x, prev_y = 0, 0
    smooth_x, smooth_y = 0, 0

    draw_color = (0, 0, 255)
    brush_thickness = 7
    eraser_thickness = 28
    smoothing = 3

    last_time = 0
    clear_cooldown = 0

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Could not read frame.")
            break

        frame = cv2.flip(frame, 1)

        if canvas is None:
            canvas = frame.copy() * 0

        current_time = time.time()
        timestamp_ms = int(current_time * 1000)

        frame = tracker.find_hands(frame, timestamp_ms)
        landmarks = tracker.find_position(frame, draw=True)

        mode_text = "IDLE"
        finger_text = "No fingers up"
        hand_label = "Unknown"

        if tracker.is_valid_hand(landmarks):
            hand_label = tracker.get_hand_label()
            fingers = tracker.fingers_up(landmarks, hand_label)
            up_finger_names = tracker.get_up_finger_names(fingers)

            if up_finger_names:
                finger_text = ", ".join(up_finger_names)

            # index fingertip
            x, y = landmarks[8][1], landmarks[8][2]

            if smooth_x == 0 and smooth_y == 0:
                smooth_x, smooth_y = x, y

            smooth_x = int(smooth_x + (x - smooth_x) / smoothing)
            smooth_y = int(smooth_y + (y - smooth_y) / smoothing)

            cv2.circle(frame, (smooth_x, smooth_y), 8, (0, 0, 255), cv2.FILLED)

            # DRAW MODE -> Index + Middle
            if fingers == [0, 1, 1, 0, 0]:
                mode_text = "DRAW"

                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = smooth_x, smooth_y

                cv2.line(
                    canvas,
                    (prev_x, prev_y),
                    (smooth_x, smooth_y),
                    draw_color,
                    brush_thickness,
                    cv2.LINE_AA
                )
                cv2.circle(canvas, (smooth_x, smooth_y), brush_thickness // 2, draw_color, cv2.FILLED)
                cv2.circle(canvas, (prev_x, prev_y), brush_thickness // 2, draw_color, cv2.FILLED)

                prev_x, prev_y = smooth_x, smooth_y

            # AIM MODE -> Index only
            elif fingers == [0, 1, 0, 0, 0]:
                mode_text = "AIM"
                prev_x, prev_y = 0, 0

            # ERASE MODE -> Middle only
            elif fingers == [0, 0, 1, 0, 0]:
                mode_text = "ERASE"

                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = smooth_x, smooth_y

                cv2.line(
                    canvas,
                    (prev_x, prev_y),
                    (smooth_x, smooth_y),
                    (0, 0, 0),
                    eraser_thickness,
                    cv2.LINE_AA
                )
                cv2.circle(canvas, (smooth_x, smooth_y), eraser_thickness // 2, (0, 0, 0), cv2.FILLED)
                cv2.circle(frame, (smooth_x, smooth_y), eraser_thickness // 2, (255, 255, 255), 2)

                prev_x, prev_y = smooth_x, smooth_y

            # CLEAR MODE -> All fingers up
            elif fingers == [1, 1, 1, 1, 1]:
                mode_text = "CLEAR"

                if current_time - clear_cooldown > 1.0:
                    canvas = frame.copy() * 0
                    clear_cooldown = current_time

                prev_x, prev_y = 0, 0

            else:
                prev_x, prev_y = 0, 0

        else:
            prev_x, prev_y = 0, 0
            smooth_x, smooth_y = 0, 0

        gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        frame = cv2.bitwise_and(frame, mask)
        frame = cv2.bitwise_or(frame, canvas)

        fps = 0
        if last_time != 0:
            fps = 1 / (current_time - last_time)
        last_time = current_time

        cv2.rectangle(frame, (10, 10), (560, 185), (20, 20, 20), -1)

        cv2.putText(
            frame,
            f"Hand: {hand_label}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Up Fingers: {finger_text}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Mode: {mode_text}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (430, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2
        )

        cv2.putText(
            frame,
            "Index=Aim | Index+Middle=Draw | Middle=Erase",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "All fingers=Clear | C=Clear | Q=Quit",
            (20, 172),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2
        )

        cv2.imshow("AirAuth-SOC", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas = frame.copy() * 0
            prev_x, prev_y = 0, 0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
