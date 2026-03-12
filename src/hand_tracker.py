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

                if draw and idx == 8:
                    cv2.circle(frame, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

            if draw:
                connections = self.vision.HandLandmarksConnections.HAND_CONNECTIONS
                for connection in connections:
                    start_idx = connection.start
                    end_idx = connection.end

                    x1, y1 = landmark_list[start_idx][1], landmark_list[start_idx][2]
                    x2, y2 = landmark_list[end_idx][1], landmark_list[end_idx][2]

                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return landmark_list
