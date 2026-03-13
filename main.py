import cv2
import time
import os
from datetime import datetime
from src.hand_tracker import HandTracker


def main():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()

    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    cap.set(3, 640)
    cap.set(4, 480)

    canvas = None
    prev_x, prev_y = 0, 0
    smooth_x, smooth_y = 0, 0

    brush_thickness = 7
    eraser_thickness = 30
    smoothing = 3

    colors = [
        (0, 0, 255),    # Red
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 255, 255),  # Yellow
        (255, 255, 255) # White
    ]
    color_names = ["RED", "BLUE", "GREEN", "YELLOW", "WHITE"]

    color_boxes = [
        (0, 128),
        (128, 256),
        (256, 384),
        (384, 512),
        (512, 640)
    ]

    color_index = 0
    draw_color = colors[color_index]

    locked = False
    save_count = 0

    last_time = 0
    last_color_change_time = 0
    last_save_time = 0
    last_lock_toggle_time = 0
    last_clear_time = 0

    color_cooldown = 0.35
    save_cooldown = 1.0
    lock_cooldown = 0.8
    clear_cooldown = 0.8

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
        landmarks = tracker.find_position(frame, True)

        mode_text = "IDLE"
        hand_label = "Unknown"
        finger_text = "No fingers up"

        if tracker.is_valid_hand(landmarks):
            hand_label = tracker.get_hand_label()
            fingers = tracker.fingers_up(landmarks, hand_label)
            up_finger_names = tracker.get_up_finger_names(fingers)

            if up_finger_names:
                finger_text = ", ".join(up_finger_names)

            x, y = landmarks[8][1], landmarks[8][2]

            if smooth_x == 0 and smooth_y == 0:
                smooth_x, smooth_y = x, y

            smooth_x = int(smooth_x + (x - smooth_x) / smoothing)
            smooth_y = int(smooth_y + (y - smooth_y) / smoothing)

            cv2.circle(frame, (smooth_x, smooth_y), 8, (0, 0, 255), cv2.FILLED)

            # COLOR SELECT -> only Index finger + top bar + cooldown
            if fingers == [0, 1, 0, 0, 0] and smooth_y < 60:
                mode_text = "COLOR SELECT"

                if current_time - last_color_change_time > color_cooldown:
                    for i, (start_x, end_x) in enumerate(color_boxes):
                        if start_x <= smooth_x < end_x:
                            color_index = i
                            draw_color = colors[color_index]
                            last_color_change_time = current_time
                            break

                prev_x, prev_y = 0, 0

            # DRAW -> Index + Middle
            elif fingers == [0, 1, 1, 0, 0] and not locked:
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

            # AIM -> Index only
            elif fingers == [0, 1, 0, 0, 0]:
                mode_text = "AIM"
                prev_x, prev_y = 0, 0

            # ERASE -> Middle only
            elif fingers == [0, 0, 1, 0, 0] and not locked:
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

            # CLEAR -> All fingers
            elif fingers == [1, 1, 1, 1, 1]:
                mode_text = "CLEAR"

                if current_time - last_clear_time > clear_cooldown:
                    canvas = frame.copy() * 0
                    last_clear_time = current_time

                prev_x, prev_y = 0, 0

            # SAVE -> Pinky only
            elif fingers == [0, 0, 0, 0, 1]:
                mode_text = "SAVE"

                if current_time - last_save_time > save_cooldown:
                    os.makedirs("screenshots", exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"screenshots/airauth_{timestamp}_{save_count}.png"
                    cv2.imwrite(filename, canvas)
                    save_count += 1
                    last_save_time = current_time

                prev_x, prev_y = 0, 0

            # LOCK / UNLOCK -> Thumb + Pinky
            elif fingers == [1, 0, 0, 0, 1]:
                mode_text = "LOCK/UNLOCK"

                if current_time - last_lock_toggle_time > lock_cooldown:
                    locked = not locked
                    last_lock_toggle_time = current_time

                prev_x, prev_y = 0, 0

            else:
                prev_x, prev_y = 0, 0
        else:
            prev_x, prev_y = 0, 0
            smooth_x, smooth_y = 0, 0

        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        frame = cv2.bitwise_and(frame, mask)
        frame = cv2.bitwise_or(frame, canvas)

        # Top color bar
        for i, c in enumerate(colors):
            start_x, end_x = color_boxes[i]
            cv2.rectangle(frame, (start_x, 0), (end_x, 60), c, -1)

            if i == color_index:
                cv2.rectangle(frame, (start_x, 0), (end_x, 60), (255, 255, 255), 3)

        current = time.time()
        fps = 1 / (current - last_time) if last_time != 0 else 0
        last_time = current

        cv2.rectangle(frame, (10, 70), (620, 190), (20, 20, 20), -1)

        cv2.putText(
            frame, f"Hand: {hand_label}", (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

        cv2.putText(
            frame, f"Up Fingers: {finger_text}", (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 0), 2
        )

        cv2.putText(
            frame, f"Mode: {mode_text}", (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )

        cv2.putText(
            frame, f"Color: {color_names[color_index]}", (250, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, draw_color, 2
        )

        lock_text = "LOCKED" if locked else "UNLOCKED"
        lock_color = (0, 0, 255) if locked else (0, 255, 0)
        cv2.putText(
            frame, f"State: {lock_text}", (250, 130),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, lock_color, 2
        )

        cv2.putText(
            frame, f"FPS: {int(fps)}", (500, 160),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2
        )

        cv2.putText(
            frame,
            "Index=Aim | Top Bar+Index=Color | Index+Middle=Draw",
            (10, 445),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2
        )

        cv2.putText(
            frame,
            "Middle=Erase | Pinky=Save | Thumb+Pinky=Lock | All=Clear | Q=Quit",
            (10, 470),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2
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
