import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandTracker:
    def __init__(self, model_path="assets/hand_landmarker.task", num_hands=1):
        self.mp = mp
        self.vision = vision
        self.running_mode = vision.RunningMode.VIDEO

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=self.running_mode,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.detector = vision.HandLandmarker.create_from_options(options)
        self.last_result = None
        self.tip_ids = [4, 8, 12, 16, 20]
        self.tip_names = {
            4: "Thumb",
            8: "Index",
            12: "Middle",
            16: "Ring",
            20: "Pinky"
        }

    def find_hands(self, frame, timestamp_ms):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        self.last_result = self.detector.detect_for_video(mp_image, timestamp_ms)
        return frame

    def find_position(self, frame, draw=True):
        landmark_list = []

        if self.last_result and self.last_result.hand_landmarks:
            h, w, _ = frame.shape
            hand_landmarks = self.last_result.hand_landmarks[0]

            for idx, lm in enumerate(hand_landmarks):
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmark_list.append((idx, cx, cy))

                if draw and idx in self.tip_ids:
                    cv2.circle(frame, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                    cv2.putText(
                        frame,
                        self.tip_names[idx],
                        (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1
                    )

            if draw:
                connections = self.vision.HandLandmarksConnections.HAND_CONNECTIONS
                for connection in connections:
                    start_idx = connection.start
                    end_idx = connection.end

                    x1, y1 = landmark_list[start_idx][1], landmark_list[start_idx][2]
                    x2, y2 = landmark_list[end_idx][1], landmark_list[end_idx][2]

                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return landmark_list

    def get_hand_label(self):
        if self.last_result and self.last_result.handedness:
            category = self.last_result.handedness[0][0]
            return category.category_name
        return "Unknown"

    def fingers_up(self, landmarks, hand_label="Right"):
        if not landmarks:
            return []

        fingers = []

        # Thumb logic changes for left/right hand
        if hand_label == "Right":
            fingers.append(1 if landmarks[4][1] < landmarks[3][1] else 0)
        else:
            fingers.append(1 if landmarks[4][1] > landmarks[3][1] else 0)

        # Index, Middle, Ring, Pinky
        for tip_id in [8, 12, 16, 20]:
            fingers.append(1 if landmarks[tip_id][2] < landmarks[tip_id - 2][2] else 0)

        return fingers

    def get_up_finger_names(self, fingers):
        names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        up_names = []

        for i in range(len(fingers)):
            if fingers[i] == 1:
                up_names.append(names[i])

        return up_names
