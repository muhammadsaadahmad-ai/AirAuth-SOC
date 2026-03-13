import os
import cv2
from datetime import datetime


class CanvasManager:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.canvas = None

    def initialize(self, frame):
        if self.canvas is None:
            self.canvas = frame.copy() * 0

    def get_canvas(self):
        return self.canvas

    def clear(self, frame):
        self.canvas = frame.copy() * 0

    def draw_line(self, x1, y1, x2, y2, color, thickness):
        cv2.line(
            self.canvas,
            (x1, y1),
            (x2, y2),
            color,
            thickness,
            cv2.LINE_AA
        )
        cv2.circle(self.canvas, (x2, y2), thickness // 2, color, cv2.FILLED)
        cv2.circle(self.canvas, (x1, y1), thickness // 2, color, cv2.FILLED)

    def erase_line(self, x1, y1, x2, y2, thickness):
        erase_color = (0, 0, 0)
        cv2.line(
            self.canvas,
            (x1, y1),
            (x2, y2),
            erase_color,
            thickness,
            cv2.LINE_AA
        )
        cv2.circle(self.canvas, (x2, y2), thickness // 2, erase_color, cv2.FILLED)

    def save_screenshot(self, save_count, folder="screenshots"):
        os.makedirs(folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(folder, f"airauth_{timestamp}_{save_count}.png")
        cv2.imwrite(filename, self.canvas)
        return filename

    def save_project_snapshot(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"final_canvas_{timestamp}.png"
        cv2.imwrite(filename, self.canvas)
        return filename

    def merge_with_frame(self, frame):
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        merged = cv2.bitwise_and(frame, mask)
        merged = cv2.bitwise_or(merged, self.canvas)
        return merged
