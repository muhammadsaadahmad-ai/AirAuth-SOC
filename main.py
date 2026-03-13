import cv2
import time

from src.hand_tracker import HandTracker
from src.canvas_manager import CanvasManager
from src.gesture_logic import GestureLogic
from src.auth_manager import AuthManager
from src.utils import ActionLogger


def format_finger_text(fingers, up_finger_names):
    count = sum(fingers)

    if count == 0:
        return "No fingers up"
    if count == 5:
        return "All 5 fingers"
    if count == 1:
        return up_finger_names[0]
    if count == 2:
        return ", ".join(up_finger_names)
    return f"{count} fingers up"


def main():
    auth_manager = AuthManager()

    if not auth_manager.login():
        return

    cap = cv2.VideoCapture(0)
    tracker = HandTracker()
    canvas_manager = CanvasManager()
    gesture_logic = GestureLogic()
    logger = ActionLogger()

    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    cap.set(3, 640)
    cap.set(4, 480)

    window_name = "AirAuth-SOC"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    is_fullscreen = True

    prev_x, prev_y = 0, 0
    smooth_x, smooth_y = 0, 0

    brush_thickness = 7
    eraser_thickness = 30
    smoothing = 3

    color_index = 0
    draw_color = gesture_logic.colors[color_index]

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

    should_ask_save_on_exit = False

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Could not read frame.")
            break

        frame = cv2.flip(frame, 1)
        canvas_manager.initialize(frame)

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
            finger_text = format_finger_text(fingers, up_finger_names)

            x, y = landmarks[8][1], landmarks[8][2]

            if smooth_x == 0 and smooth_y == 0:
                smooth_x, smooth_y = x, y

            smooth_x = int(smooth_x + (x - smooth_x) / smoothing)
            smooth_y = int(smooth_y + (y - smooth_y) / smoothing)

            cv2.circle(frame, (smooth_x, smooth_y), 8, (0, 0, 255), cv2.FILLED)

            mode = gesture_logic.get_mode(
                fingers,
                auth_manager.is_locked(),
                smooth_y
            )
            mode_text = mode

            if mode == "COLOR_SELECT":
                if current_time - last_color_change_time > color_cooldown:
                    selected_index, selected_color, _ = gesture_logic.get_selected_color(smooth_x)
                    if selected_index is not None:
                        color_index = selected_index
                        draw_color = selected_color
                        last_color_change_time = current_time
                        logger.log(
                            "COLOR_CHANGED",
                            f"color={gesture_logic.color_names[color_index]}"
                        )
                prev_x, prev_y = 0, 0

            elif mode == "DRAW":
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = smooth_x, smooth_y

                canvas_manager.draw_line(
                    prev_x, prev_y,
                    smooth_x, smooth_y,
                    draw_color,
                    brush_thickness
                )
                prev_x, prev_y = smooth_x, smooth_y

            elif mode == "AIM":
                prev_x, prev_y = 0, 0

            elif mode == "ERASE":
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = smooth_x, smooth_y

                canvas_manager.erase_line(
                    prev_x, prev_y,
                    smooth_x, smooth_y,
                    eraser_thickness
                )
                cv2.circle(
                    frame,
                    (smooth_x, smooth_y),
                    eraser_thickness // 2,
                    (255, 255, 255),
                    2
                )
                prev_x, prev_y = smooth_x, smooth_y

            elif mode == "CLEAR":
                if current_time - last_clear_time > clear_cooldown:
                    canvas_manager.clear(frame)
                    last_clear_time = current_time
                    logger.log("CANVAS_CLEARED", "gesture=all_fingers")
                prev_x, prev_y = 0, 0

            elif mode == "SAVE":
                if current_time - last_save_time > save_cooldown:
                    filename = canvas_manager.save_screenshot(save_count)
                    logger.log("SCREENSHOT_SAVED", f"file={filename}")
                    save_count += 1
                    last_save_time = current_time
                prev_x, prev_y = 0, 0

            elif mode == "LOCK_TOGGLE":
                if current_time - last_lock_toggle_time > lock_cooldown:
                    auth_manager.toggle_lock()
                    last_lock_toggle_time = current_time
                    state = auth_manager.get_state_text()
                    logger.log("LOCK_STATE_CHANGED", f"state={state}")
                prev_x, prev_y = 0, 0

            else:
                prev_x, prev_y = 0, 0

        else:
            prev_x, prev_y = 0, 0
            smooth_x, smooth_y = 0, 0

        frame = canvas_manager.merge_with_frame(frame)

        for i, c in enumerate(gesture_logic.colors):
            start_x, end_x = gesture_logic.color_boxes[i]
            cv2.rectangle(frame, (start_x, 0), (end_x, 60), c, -1)

            if i == color_index:
                cv2.rectangle(frame, (start_x, 0), (end_x, 60), (255, 255, 255), 3)

        current = time.time()
        fps = 1 / (current - last_time) if last_time != 0 else 0
        last_time = current

        # Bigger HUD panel
        cv2.rectangle(frame, (10, 75), (630, 230), (20, 20, 20), -1)

        # Left column
        cv2.putText(
            frame,
            f"Hand: {hand_label}",
            (25, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Fingers: {finger_text}",
            (25, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Mode: {mode_text}",
            (25, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 0, 255),
            2
        )

        # Right column
        cv2.putText(
            frame,
            f"Color: {gesture_logic.color_names[color_index]}",
            (340, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            draw_color,
            2
        )

        cv2.putText(
            frame,
            f"State: {auth_manager.get_state_text()}",
            (340, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            auth_manager.get_state_color(),
            2
        )

        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (340, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (200, 200, 200),
            2
        )

        cv2.putText(
            frame,
            "Index=Aim | Top Bar+Index=Color | Index+Middle=Draw",
            (10, 445),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Middle=Erase | Pinky=Save | Thumb+Pinky=Lock | All=Clear",
            (10, 470),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Q=Quit | F=Fullscreen Toggle",
            (10, 495),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2
        )

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            should_ask_save_on_exit = True
            break

        elif key == ord('c'):
            canvas_manager.clear(frame)
            prev_x, prev_y = 0, 0
            logger.log("CANVAS_CLEARED", "gesture=keyboard_c")

        elif key == ord('f'):
            if is_fullscreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                is_fullscreen = False
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                is_fullscreen = True

    cap.release()
    cv2.destroyAllWindows()

    if should_ask_save_on_exit:
        choice = input("\nDo you want to save the current canvas before exit? (y/n): ").strip().lower()
        if choice == 'y':
            filename = canvas_manager.save_project_snapshot()
            logger.log("EXIT_SAVE", f"file={filename}")
            print(f"Canvas saved as: {filename}")
        else:
            print("Canvas not saved.")


if __name__ == "__main__":
    main()
