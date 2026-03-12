import cv2
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
        frame = tracker.find_hands(frame)
        landmarks = tracker.find_position(frame)

        if landmarks:
            _, x, y = landmarks[8]
            cv2.putText(
                frame,
                f"Index Finger: ({x}, {y})",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
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
