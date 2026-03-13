import cv2
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandTracker:
    def __init__(self, model_path="assets/hand_landmarker.task", num_hands=1):
        self.mp = mp
        self.vision = vision

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.detector = vision.HandLandmarker.create_from_options(options)
        self.last_result = None
        self.tip_ids = [4, 8, 12, 16, 20]

    def find_hands(self, frame, timestamp_ms):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB,
            data=rgb
        )
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

            if draw and len(landmark_list) == 21:
                for idx in self.tip_ids:
                    cx, cy = landmark_list[idx][1], landmark_list[idx][2]
                    cv2.circle(frame, (cx, cy), 7, (255, 0, 255), cv2.FILLED)

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
            label = self.last_result.handedness[0][0].category_name

            # webcam frame flipped
            if label == "Left":
                return "Right"
            elif label == "Right":
                return "Left"

            return label

        return "Unknown"

    def is_valid_hand(self, landmarks):
        if not landmarks or len(landmarks) != 21:
            return False

        wrist_x, wrist_y = landmarks[0][1], landmarks[0][2]
        middle_mcp_x, middle_mcp_y = landmarks[9][1], landmarks[9][2]

        distance = abs(wrist_x - middle_mcp_x) + abs(wrist_y - middle_mcp_y)
        return distance > 25

    def fingers_up(self, landmarks, hand_label="Right"):
        if not self.is_valid_hand(landmarks):
            return [0, 0, 0, 0, 0]

        fingers = []

        # Thumb
        if hand_label == "Right":
            fingers.append(1 if landmarks[4][1] < landmarks[3][1] else 0)
        else:
            fingers.append(1 if landmarks[4][1] > landmarks[3][1] else 0)

        # Tolerance added to reduce false idle
        tolerance = 12

        for tip_id in [8, 12, 16, 20]:
            tip_y = landmarks[tip_id][2]
            pip_y = landmarks[tip_id - 2][2]
            fingers.append(1 if tip_y < pip_y - tolerance else 0)

        return fingers

    def get_up_finger_names(self, fingers):
        names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        up_names = []

        for i in range(len(fingers)):
            if fingers[i] == 1:
                up_names.append(names[i])

        return up_names

    def distance(self, p1, p2, landmarks):
        if len(landmarks) != 21:
            return 9999

        x1, y1 = landmarks[p1][1], landmarks[p1][2]
        x2, y2 = landmarks[p2][1], landmarks[p2][2]
        return math.hypot(x2 - x1, y2 - y1)

    def is_pinching(self, landmarks, threshold=35):
        # thumb tip = 4, index tip = 8
        return self.distance(4, 8, landmarks) < threshold
