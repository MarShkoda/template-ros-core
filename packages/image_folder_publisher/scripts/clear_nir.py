import cv2
import numpy as np
from enum import Enum

class ColorLine(Enum):
    YELLOW = 1
    WHITE = 2


class FastLaneDetector:
    def __init__(self, width, height, color: ColorLine):
        self.width = width
        self.height = height
        self.color = color

        # ROI
        self.y_top = int(height * 0.55)
        self.y_bottom = int(height * 0.90)

        # HSV masks
        if color == ColorLine.YELLOW:
            self.lower = np.array([20, 80, 80])
            self.upper = np.array([40, 255, 255])
        else:
            self.lower = np.array([0, 0, 200])
            self.upper = np.array([180, 40, 255])

    def process(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, self.lower, self.upper)

        roi = mask[self.y_top:self.y_bottom, :]

        hist = np.sum(roi, axis=0)

        if np.max(hist) < 50:
            return None, 0.0

        xs = np.arange(self.width)
        center_x = np.sum(xs * hist) / np.sum(hist)

        robot_center = self.width / 2
        error = (center_x - robot_center) / robot_center

        return center_x, error

    def draw(self, image, center_x):
        if center_x is None:
            return image

        x = int(center_x)
        cv2.line(image, (x, self.y_top), (x, self.y_bottom), (0,255,0), 2)
        cv2.rectangle(image, (0, self.y_top), (self.width, self.y_bottom), (255,0,0), 2)
        return image