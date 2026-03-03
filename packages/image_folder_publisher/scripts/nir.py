import cv2
import numpy as np
import enum
import math
import os
import matplotlib.pyplot as plt

@enum.unique
class ColorLine(enum.Enum):
    yellow = 2
    white = 1


class ImageLineProcessing:
    def __init__(self, image, width, height, color: ColorLine):
        self.image = image #cv2.GaussianBlur(image, (5, 5), 0)
        self.width = width
        self.height = height
        self.color = color

        # Цветовая маска (BGR, Duckietown)
        if color == ColorLine.yellow:
            lower = np.array([175, 165, 20], dtype=np.uint8)
            upper = np.array([250, 250, 150], dtype=np.uint8)
        else:
            lower = np.array([160, 160, 160], dtype=np.uint8)
            upper = np.array([255, 255, 255], dtype=np.uint8)

        self.color_mask = cv2.inRange(self.image, lower, upper)

    # --------------------------------------------------
    # ROI
    # --------------------------------------------------
    def region_selection(self):
        mask = np.zeros_like(self.color_mask)

        rows, cols = self.color_mask.shape[:2]

        if self.color == ColorLine.yellow:
            left, right = 0.0, 0.8
        else:
            left, right = 0.2, 1.0

        vertices = np.array([[
            (cols * left,  rows * 0.8),
            (cols * left,  rows * 0.1),
            (cols * right, rows * 0.1),
            (cols * right, rows * 0.8),
        ]], dtype=np.int32)

        cv2.fillPoly(mask, vertices, 255)
        return cv2.bitwise_and(self.color_mask, mask)

    # --------------------------------------------------
    # Детекция линии (Hough)
    # --------------------------------------------------
    def detect_line(self, binary):
        edges = cv2.Canny(binary, 80, 160)

        params = dict(
            threshold=20,
            minLineLength=30,
            maxLineGap=10
        )

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, **params)
        if lines is None:
            return None

        # Фильтрация почти горизонтальных линий
        candidates = []
        for l in lines:
            x1, y1, x2, y2 = l[0]
            if abs(x2 - x1) < 5:   # почти вертикальная — ок
                candidates.append((x1, y1, x2, y2))
            else:
                slope = abs((y2 - y1) / (x2 - x1))
                if slope > 0.3:
                    candidates.append((x1, y1, x2, y2))

        if not candidates:
            return None

        # Усреднение по всем сегментам
        xs, ys = [], []
        for x1, y1, x2, y2 in candidates:
            xs += [x1, x2]
            ys += [y1, y2]

        if len(xs) < 2:
            return None

        # Линейная регрессия x(y)
        fit = np.polyfit(ys, xs, 1)
        k, b = fit

        y_bottom = int(self.height * 0.78)
        y_top = int(self.height * 0.55)

        x_bottom = int(k * y_bottom + b)
        x_top = int(k * y_top + b)

        return (x_bottom, y_bottom, x_top, y_top), k, b

    # --------------------------------------------------
    # Основной метод
    # --------------------------------------------------
    def process(self):
        roi = self.region_selection()
        _, binary = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)

        result = self.detect_line(binary)
        if result is None:
            return {
                "valid": False,
                "line": None,
                "k": None,
                "b": None
            }

        line, k, b = result
        return {
            "valid": True,
            "line": line,
            "k": k,
            "b": b
        }

    def x_at_y(self, model, y):
        if not model["valid"]:
            return None
        return model["k"] * y + model["b"]

    def draw(self, image, model, color=(0, 255, 0), thickness=2):
        if not model["valid"]:
            return image
        x1, y1, x2, y2 = model["line"]
        return cv2.line(image.copy(), (x1, y1), (x2, y2), color, thickness)


def get_bird_eye_transform():
    CAMERA_FOV_Y = 75.0  # градусы
    CAMERA_ANGLE = 19.15 * math.pi / 180.0  # радианы
    CAMERA_FLOOR_DIST = 0.108  # м

    DEFAULT_CAMERA_WIDTH = 640
    DEFAULT_CAMERA_HEIGHT = 480

    w = DEFAULT_CAMERA_WIDTH
    h = DEFAULT_CAMERA_HEIGHT

    # Вертикальное и горизонтальное поле зрения
    fov_y = math.radians(CAMERA_FOV_Y)
    fov_x = 2 * math.atan(math.tan(fov_y / 2) * (w / h))

    # Фокальные расстояния в пикселях
    fx = w / (2 * math.tan(fov_x / 2))
    fy = h / (2 * math.tan(fov_y / 2))

    # Центр проекции
    cx, cy = w / 2, h / 2

    # 4 точки в нижней части изображения (трапеция дороги)
    img_pts = np.float32([
        [w*0.2, h*0.9],  # левый низ
        [w*0.8, h*0.9],  # правый низ
        [w*0.45, h*0.6], # левый верх
        [w*0.55, h*0.6]  # правый верх
    ])

    def img_to_ground(u, v):
        # нормализованные координаты
        x_cam = (u - cx) / fx
        y_cam = (v - cy) / fy

        # направляющий вектор луча в системе камеры
        ray = np.array([x_cam, 1.0, y_cam])  # (x, forward, y)

        # поворот камеры вниз
        R = np.array([
            [1, 0, 0],
            [0,  math.cos(-CAMERA_ANGLE), -math.sin(-CAMERA_ANGLE)],
            [0,  math.sin(-CAMERA_ANGLE),  math.cos(-CAMERA_ANGLE)]
        ])
        ray = R @ ray

        # пересечение с плоскостью z=0
        t = CAMERA_FLOOR_DIST / (-ray[2])
        X = ray[0]*t
        Y = ray[1]*t
        return [X, Y]

    world_pts = np.float32([img_to_ground(u, v) for (u, v) in img_pts])

    minx, miny = np.min(world_pts, axis=0)
    maxx, maxy = np.max(world_pts, axis=0)

    world_norm = (world_pts - [minx, miny]) / ([maxx-minx, maxy-miny])
    world_norm[:,0] *= w
    world_norm[:,1] *= h

    M = cv2.getPerspectiveTransform(img_pts, world_norm)
    return M

DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480
PI = 3.1415926

alpha = (13 - 90) * PI / 180
beta = (89 - 90) * PI / 180
gamma = (95 - 90) * PI / 180
focalLength = 653 #cv2.getTrackbarPos("f", "Result")
dist = 666 #cv2.getTrackbarPos("Distance", "Result")

image_size = (DEFAULT_CAMERA_WIDTH, DEFAULT_CAMERA_HEIGHT)
w, h = image_size

A1 = np.array([[1, 0, -w / 2],
            [0, 1, -h / 2],
            [0, 0, 0],
            [0, 0, 1]], dtype=np.float32)

RX = np.array([[1, 0, 0, 0],
            [0, math.cos(alpha), -math.sin(alpha), 0],
            [0, math.sin(alpha), math.cos(alpha), 0],
            [0, 0, 0, 1]], dtype=np.float32)

RY = np.array([[math.cos(beta), 0, -math.sin(beta), 0],
            [0, 1, 0, 0],
            [math.sin(beta), 0, math.cos(beta), 0],
            [0, 0, 0, 1]], dtype=np.float32)

RZ = np.array([[math.cos(gamma), -math.sin(gamma), 0, 0],
            [math.sin(gamma), math.cos(gamma), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]], dtype=np.float32)

R = np.dot(np.dot(RX, RY), RZ)

T = np.array([[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, dist],
            [0, 0, 0, 1]], dtype=np.float32)

K = np.array([[focalLength, 0, w / 2, 0],
            [0, focalLength, h / 2, 0],
            [0, 0, 1, 0]], dtype=np.float32)

transformationMat = np.dot(np.dot(np.dot(K, T), R), A1)

def warp_bird_eye(image):
    destination = cv2.warpPerspective(image, transformationMat, image_size, flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
    return destination

import time
import math

class PIDController:
    def __init__(self,
                 kp_horizontal, ki_horizontal, kd_horizontal,
                 kp_angular, ki_angular, kd_angular,
                 max_linear_velocity=0.5,
                 max_angular_velocity=2.0):
        """Initializes the PID controller.

        Args:
            kp_horizontal (float): Proportional gain for horizontal error.
            ki_horizontal (float): Integral gain for horizontal error.
            kd_horizontal (float): Derivative gain for horizontal error.
            kp_angular (float): Proportional gain for angular error.
            ki_angular (float): Integral gain for angular error.
            kd_angular (float): Derivative gain for angular error.
            max_linear_velocity (float): Maximum allowed linear velocity (e.g., m/s).
            max_angular_velocity (float): Maximum allowed angular velocity (e.g., rad/s).
        """
        self.kp_h = kp_horizontal
        self.ki_h = ki_horizontal
        self.kd_h = kd_horizontal

        self.kp_a = kp_angular
        self.ki_a = ki_angular
        self.kd_a = kd_angular

        self.max_linear_v = max_linear_velocity
        self.max_angular_v = max_angular_velocity

        # Internal state for horizontal error PID
        self.prev_error_h = 0.0
        self.integral_h = 0.0

        # Internal state for angular error PID
        self.prev_error_a = 0.0
        self.integral_a = 0.0

        self.last_time = None

    def update(self, horizontal_error, angular_error):
        """Updates the PID controller with new errors and computes control outputs.

        Args:
            horizontal_error (float): The horizontal deviation from the desired path (e.g., in pixels).
                                      Positive if lane center is to the right of the robot's center.
            angular_error (float): The angular deviation from the desired heading (e.g., in radians).
                                   Positive if the lane is slanting to the right relative to robot's heading.

        Returns:
            tuple: (linear_velocity, angular_velocity) in the robot's units.
        """
        current_time = time.time()
        if self.last_time is None:
            # First call, initialize and return base velocities
            self.last_time = current_time
            self.prev_error_h = horizontal_error
            self.prev_error_a = angular_error
            return self.max_linear_v, 0.0

        dt = current_time - self.last_time
        if dt <= 0:
            # Avoid division by zero or negative time step
            return self.max_linear_v, 0.0

        # PID for Horizontal Error (influences angular velocity)
        self.integral_h += horizontal_error * dt
        derivative_h = (horizontal_error - self.prev_error_h) / dt
        output_h = self.kp_h * horizontal_error + self.ki_h * self.integral_h + self.kd_h * derivative_h
        self.prev_error_h = horizontal_error

        # PID for Angular Error (influences angular velocity)
        self.integral_a += angular_error * dt
        derivative_a = (angular_error - self.prev_error_a) / dt
        output_a = self.kp_a * angular_error + self.ki_a * self.integral_a + self.kd_a * derivative_a
        self.prev_error_a = angular_error

        # Combine outputs to determine angular velocity
        # Convention: Positive angular_velocity for left turn, negative for right turn.
        # - A positive horizontal_error (lane right) requires a right turn (negative omega).
        # - A positive angular_error (lane slanting right) requires a right turn (negative omega).
        angular_velocity = -output_h - output_a

        # Constrain angular velocity to defined limits
        angular_velocity = max(-self.max_angular_v, min(self.max_angular_v, angular_velocity))

        # Linear velocity (can be constant or reduced based on angular_velocity)
        linear_velocity = self.max_linear_v
        # Optional: reduce linear velocity for sharp turns
        # linear_velocity = self.max_linear_v * (1 - abs(angular_velocity) / (2 * self.max_angular_v))
        # linear_velocity = max(0.0, min(self.max_linear_v, linear_velocity))

        self.last_time = current_time
        return linear_velocity, angular_velocity

P = 1/30
I = 1/20
D = 1/20
pid_controller_old = PIDController(
    kp_horizontal=0.02, ki_horizontal=0.0, kd_horizontal=0.001, # Example gains for horizontal control
    kp_angular=0.5, ki_angular=0.0, kd_angular=0.1,             # Example gains for angular control
    max_linear_velocity=0.3, # Max linear velocity in m/s (adjust for Duckietown)
    max_angular_velocity=2.0 # Max angular velocity in rad/s (adjust for Duckietown)
)

pid_controller = PIDController(
    kp_horizontal=0.008,      # Уменьшил пропорциональную составляющую
    ki_horizontal=0.0001,      # Добавил небольшую интегральную
    kd_horizontal=0.0005,      # Уменьшил дифференциальную
    
    kp_angular=0.3,           # Уменьшил для более плавного управления
    ki_angular=0.0001,        # Добавил интегральную для устранения статической ошибки
    kd_angular=0.05,          # Уменьшил дифференциальную
    
    max_linear_velocity=0.25,  # Немного уменьшил скорость
    max_angular_velocity=1.5   # Уменьшил максимальную угловую скорость
)
horizontal_error = 0.0
angular_error = 0.0
linear_velocity = 0.0
angular_velocity = 0.0

