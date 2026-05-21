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

'''colors:
  RED:
    low_1: [0,140,100]
    high_1: [15,255,255]
    low_2: [165,140,100]
    high_2: [180,255,255]
  WHITE:
    low: [0,0,150]
    high: [180,100,255]
  YELLOW:
    low: [25,140,100]
    high: [45,255,255]
'''



import enum
import cv2
import numpy as np


@enum.unique
class ColorLine(enum.Enum):
    yellow = 2
    white = 1


# ==========================================================
# BASE
# ==========================================================
class BaseImageLineProcessing:
    def __init__(self, width, height, color: 'ColorLine', debug=False):
        self.width = width
        self.height = height
        self.color = color
        self.debug = debug

        if color == ColorLine.yellow:
            self.lower = np.array([20, 100, 100], np.uint8)
            self.upper = np.array([30, 255, 255], np.uint8)
            self.left, self.right = 0.0, 0.8
        else:
            self.lower = np.array([0, 0, 150], np.uint8)
            self.upper = np.array([180, 100, 255], np.uint8)
            self.left, self.right = 0.6, 1.0

        self._roi_mask = None

    def _get_roi_mask(self):
        if self._roi_mask is None:
            mask = np.zeros((self.height, self.width), dtype=np.uint8)

            vertices = np.array([[
                (int(self.width * self.left),  int(self.height * 0.55)),
                (int(self.width * self.left),  int(self.height * 0.95)),
                (int(self.width * self.right), int(self.height * 0.95)),
                (int(self.width * self.right), int(self.height * 0.55)),
            ]], np.int32)

            cv2.fillPoly(mask, vertices, 255)
            self._roi_mask = mask

        return self._roi_mask

    def preprocess(self, image):
        hsv = cv2.cvtColor(cv2.blur(image, (3, 3)), cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        roi = cv2.bitwise_and(mask, self._get_roi_mask())
        _, binary = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)

        #if self.debug:
            #cv2.imshow("01_mask", mask)
            #cv2.imshow("02_roi", roi)
            #cv2.imshow("03_binary", binary)

        return binary

    def fit_result(self, pts, dbg=None):
        if len(pts) < 2:
            return {"valid": False, "line": None, "k": None, "b": None}

        pts = np.array(pts, np.float32)

        vx, vy, x0, y0 = cv2.fitLine(
            pts, cv2.DIST_L2, 0, 0.01, 0.01
        )

        vx, vy, x0, y0 = vx[0], vy[0], x0[0], y0[0]

        if abs(vy) < 1e-6:
            return {"valid": False, "line": None, "k": None, "b": None}

        k = vx / vy
        b = x0 - k * y0

        y1 = int(self.height * 0.78)
        y2 = int(self.height * 0.55)

        x1 = int(k * y1 + b)
        x2 = int(k * y2 + b)

        if self.debug and dbg is not None:
            cv2.line(dbg, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.imshow("99_result", dbg)
            cv2.waitKey(1)

        return {
            "valid": True,
            "line": (x1, y1, x2, y2),
            "k": float(k),
            "b": float(b)
        }
        
    def x_at_y(self, model, y):
        if not model["valid"]:
            return None
        return model["k"] * y + model["b"]

    def draw(self, image, model, color=(0, 255, 0), lane_center=0, center_x=0, thickness=2):
        if not model["valid"]:
            return image
            
        if (lane_center!=0 and center_x!=0):
            cv2.line(image, (lane_center,0), (lane_center,400), (255,0,255), thickness)
            cv2.line(image, (center_x,0), (center_x,400), (0,0,255), thickness)

        x1, y1, x2, y2 = model["line"]
        cv2.line(image, (x1, y1), (x2, y2), color, thickness)
        return image


# ==========================================================
# 1. HOUGH
# ==========================================================
class HoughLineProcessing(BaseImageLineProcessing):

    def process(self, image):
        binary = self.preprocess(image)
        dbg = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        lines = cv2.HoughLinesP(
            binary,
            rho=1,
            theta=np.pi / 180,
            threshold=20,
            minLineLength=20,
            maxLineGap=15
        )

        if lines is None:
            if self.debug:
                cv2.imshow("10_hough_lines", dbg)
                cv2.waitKey(1)
            return {"valid": False, "line": None, "k": None, "b": None}

        pts = []

        for l in lines:
            x1, y1, x2, y2 = l[0]
            pts.append((x1, y1))
            pts.append((x2, y2))

            if self.debug:
                cv2.line(dbg, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if self.debug:
            cv2.imshow("10_hough_lines", dbg)

        return self.fit_result(pts, dbg)


# ==========================================================
# 2. CONTOURS
# ==========================================================
class ContourLineProcessing(BaseImageLineProcessing):

    def process(self, image):
        binary = self.preprocess(image)
        dbg = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            if self.debug:
                cv2.imshow("20_contours", dbg)
                cv2.waitKey(1)
            return {"valid": False, "line": None, "k": None, "b": None}

        cnt = max(contours, key=cv2.contourArea)

        if cv2.contourArea(cnt) < 30:
            return {"valid": False, "line": None, "k": None, "b": None}

        cv2.drawContours(dbg, [cnt], -1, (255, 0, 0), 2)

        pts = cnt.reshape(-1, 2)

        if self.debug:
            cv2.imshow("20_contours", dbg)

        return self.fit_result(pts, dbg)


# ==========================================================
# 3. SLIDING WINDOW
# ==========================================================
class SlidingWindowLineProcessing(BaseImageLineProcessing):

    def process(self, image):
        binary = self.preprocess(image)
        dbg = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        histogram = np.sum(binary[binary.shape[0]//2:, :], axis=0)
        base_x = np.argmax(histogram)

        n_windows = 8
        margin = 40
        minpix = 5

        win_h = binary.shape[0] // n_windows
        current_x = base_x

        nonzero = binary.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        pts = []

        for w in range(n_windows):
            win_y_low = binary.shape[0] - (w + 1) * win_h
            win_y_high = binary.shape[0] - w * win_h

            win_x_low = current_x - margin
            win_x_high = current_x + margin

            good_inds = (
                (nonzero_y >= win_y_low) &
                (nonzero_y < win_y_high) &
                (nonzero_x >= win_x_low) &
                (nonzero_x < win_x_high)
            ).nonzero()[0]

            cv2.rectangle(
                dbg,
                (win_x_low, win_y_low),
                (win_x_high, win_y_high),
                (0, 255, 255),
                2
            )

            if len(good_inds) > minpix:
                current_x = int(np.mean(nonzero_x[good_inds]))

            pts.append((current_x, (win_y_low + win_y_high)//2))

            cv2.circle(
                dbg,
                (current_x, (win_y_low + win_y_high)//2),
                4,
                (0, 0, 255),
                -1
            )

        if self.debug:
            cv2.imshow("30_sliding_window", dbg)

        return self.fit_result(pts, dbg)

class ImageLineProcessing:
    def __init__(self, width, height, color: 'ColorLine', save_debug_dir="debug_photos"):
        self.width = width
        self.height = height
        self.color = color
        self.save_debug_dir = save_debug_dir
        # os.makedirs(self.save_debug_dir, exist_ok=True)

        if color == ColorLine.yellow:
            #self.lower = np.array([25,140,100], np.uint8)
            #self.upper = np.array([45,255,255], np.uint8)
            #self.lower = np.array([20, 80, 80])
            #self.upper = np.array([50, 255, 255])
            self.lower = np.array([20, 100, 100], np.uint8)
            self.upper = np.array([30, 255, 255], np.uint8)
            self.left, self.right = 0.0, 0.8


        else:
            self.lower = np.array([0,0,150], np.uint8)
            self.upper = np.array([180,100,255], np.uint8)
            #self.lower = np.array([0, 0, 200], dtype=np.uint8)
            #self.upper = np.array([180, 50, 255], dtype=np.uint8)
            self.left, self.right = 0.6, 1.0


        self.x1 = int(self.width * self.left)
        self.x2 = int(self.width * self.right)

        self.y1 = int(self.height * 0.35)
        self.y2 = int(self.height * 0.95)
        self._roi_mask = None
        self.image = None

    def _get_roi_mask(self, target_shape=None):
        """Create ROI mask, optionally resizing to target shape"""
        if self._roi_mask is None:
            mask = np.zeros((self.height, self.width), dtype=np.uint8)
            vertices = np.array([[
                (int(self.width * self.left),  int(self.height * 0.35)),
                (int(self.width * self.left),  int(self.height * 0.95)),
                (int(self.width * self.right), int(self.height * 0.95)),
                (int(self.width * self.right), int(self.height * 0.35)),
            ]], np.int32)
            cv2.fillPoly(mask, vertices, 255)
            self._roi_mask = mask
        
        # Resize mask if target shape is provided and different
        if target_shape is not None:
            if target_shape[:2] != self._roi_mask.shape[:2]:
                return cv2.resize(self._roi_mask, (target_shape[1], target_shape[0]))
        
        return self._roi_mask

    def region_selection(self, color_mask):
        roi_mask = self._get_roi_mask(color_mask.shape)
        return cv2.bitwise_and(color_mask, roi_mask)

    def detect_line(self, binary, orig_image=None, frame_id=None):
        debug = True #self.pub_debug.get_num_connections() > 0

        if debug:
            dbg = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        ys = np.linspace(
            int(self.height * 0.6),
            int(self.height * 0.9),
            9
        ).astype(int)

        pts = []

        for y in ys:
            row = binary[y]
            xs = np.where(row > 0)[0]
            if len(xs) < 5:
                continue
            x = int(np.median(xs))
            pts.append((x, y))
            if debug:
                cv2.circle(dbg, (x, y), 4, (0,255,255), -1)

        if len(pts) < 2:
            #if debug:
                #self.pub_debug.publish(self.bridge.cv2_to_imgmsg(dbg, "bgr8"))
            return None

        pts = np.array(pts, np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = vx[0], vy[0], x0[0], y0[0]

        if abs(vy) < 1e-5:
            return None

        y1 = int(self.height * 0.78)
        y2 = int(self.height * 0.55)
        x1 = int((y1 - y0) * vx / vy + x0)
        x2 = int((y2 - y0) * vx / vy + x0)

        if debug:
            cv2.line(dbg, (x1,y1), (x2,y2), (0,0,255), 3)
            for y in ys:
                cv2.line(dbg, (0,y), (self.width,y), (255,0,0), 1)
            k = float(vx/vy)
            angle = np.degrees(math.atan(k))
            text_y = 40
            step_y = 20
            cv2.putText(dbg, f"angle={angle:.1f}", (20,text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            text_y=text_y+step_y
            cv2.putText(dbg, f"vx={vx:.2f}", (20,text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            text_y=text_y+step_y
            cv2.putText(dbg, f"vy={vy:.1f}", (20,text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            text_y=text_y+step_y
            cv2.putText(dbg, f"k=vx/vy={k:.2f}", (20,text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            angle_rad = math.atan(k)
            text_y=text_y+step_y
            cv2.putText(dbg, f"angle_rad=math.atan(k)={angle_rad:.2f}", (20,text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            #self.pub_debug.publish(self.bridge.cv2_to_imgmsg(dbg, "bgr8"))

        # ---- Сохраняем для коллажа ----
        if orig_image is not None and frame_id is not None and frame_id < 100:
            # сохраняем в глобальный словарь
            if frame_id not in collage_store:
                collage_store[frame_id] = {}
            if self.color == ColorLine.yellow:
                collage_store[frame_id]["yellow"] = dbg
            else:
                collage_store[frame_id]["white"] = dbg
            collage_store[frame_id]["orig"] = orig_image

            # если все три изображения готовы, сохраняем коллаж
            parts = collage_store[frame_id]
            if all(k in parts for k in ["orig","yellow","white"]):
                # делаем коллаж: ширина = 3*ширина, высота = высота
                collage = np.zeros((self.height, self.width*3, 3), np.uint8)
                collage[:, :self.width] = parts["yellow"]
                collage[:, self.width:2*self.width] = parts["orig"]
                collage[:, 2*self.width:] = parts["white"]
                save_path = os.path.join(self.save_debug_dir, f"{frame_id:06d}.png")
                cv2.imwrite(save_path, collage)
                #rospy.loginfo(f"path {save_path}")
                # удаляем из словаря, чтобы не переполнять память
                del collage_store[frame_id]

        #return (x1, y1), (x2, y2)
        return (x1,y1,x2,y2), float(vx/vy), float(x0 - (vx/vy)*y0)

    def process(self, image, frame_id=None):
        roi_rgb = image[self.y1:self.y2, self.x1:self.x2]
        hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        full_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        full_mask[self.y1:self.y2, self.x1:self.x2] = mask
        result = self.detect_line(full_mask, hsv, frame_id)
        if result is None:
            return {"valid": False, "line": None, "k": None, "b": None}

        line, k, b = result
        return {"valid": True, "line": line, "k": k, "b": b}

    def x_at_y(self, model, y):
        if not model["valid"]:
            return None
        return model["k"] * y + model["b"]

    def draw(self, image, model, color=(0,255,0), lane_center=0, center_x=0, thickness=2):
        if not model["valid"]:
            return image
        x1,y1,x2,y2 = model["line"]
        if (lane_center!=0 and center_x!=0):
            cv2.line(image, (lane_center,0), (lane_center,400), (255,0,255), thickness)
            cv2.line(image, (center_x,0), (center_x,400), (0,0,255), thickness)

        return cv2.line(image, (x1,y1), (x2,y2), color, thickness)

    
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
        angular_velocity = output_h + output_a

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

pid_controller_oldik = PIDController(
    kp_horizontal=0.008,      # Уменьшил пропорциональную составляющую
    ki_horizontal=0.0001,      # Добавил небольшую интегральную
    kd_horizontal=0.0005,      # Уменьшил дифференциальную
    
    kp_angular=0.3,           # Уменьшил для более плавного управления
    ki_angular=0.0001,        # Добавил интегральную для устранения статической ошибки
    kd_angular=0.05,          # Уменьшил дифференциальную
    
    max_linear_velocity=1.5,  # Немного уменьшил скорость
    max_angular_velocity=1.5   # Уменьшил максимальную угловую скорость
)

pid_controller = PIDController(
            kp_horizontal=0.5,
            ki_horizontal=0.2,
            kd_horizontal=0.5,
            kp_angular=1.2,
            ki_angular=0.3,
            kd_angular=0.5,
            max_linear_velocity=0.3, #0.25
            max_angular_velocity=8 #1.5
        )
        
printNum = False
printPlot = True

        
pid_controller_last = PIDController(
    kp_horizontal=0.8,
    ki_horizontal=0.45 ,
    kd_horizontal=0.3,

    kp_angular=1.5,
    ki_angular=0.45,
    kd_angular=0.3,

    max_linear_velocity=0.3,
    max_angular_velocity=8.0
)
horizontal_error = 0.0
angular_error = 0.0
linear_velocity = 0.1
angular_velocity = 0.0

import numpy as np
import time

# Константы для LQR
DT = 1/30  # шаг времени
V = 0.5    # скорость (м/с)

class MyLQRController:
    """LQR контроллер для управления движением в полосе"""
    
    def __init__(self):
        self.L = 0.1  # длина колесной базы (м) - для Duckietown
        
        # Матрицы состояния для линейной модели
        # Состояние: [e, e_dot, psi, psi_dot] 
        # e - латеральное отклонение, psi - угол курса
        
        # Матрица A (системная)
        self.A = np.array([
            [1, DT, V*DT, 0],
            [0, 1, 0, V*DT],
            [0, 0, 1, DT],
            [0, 0, 0, 1]
        ])
        
        # Матрица B (управляющая) - управление: угол поворота колес (delta)
        self.B = np.array([
            [0],
            [V*DT**2/(2*self.L)],
            [0],
            [V*DT/self.L]
        ])
        
        # Матрицы весов Q и R
        # Q - веса состояний, R - вес управления
        self.Q = np.diag([10.0, 1.0, 5.0, 1.0])  # Больше вес на отклонение и угол
        self.R = np.array([[1.0]])  # Меньше вес на управление = более агрессивный контроль
        
        # Вычисляем матрицу усиления LQR
        self.K = self._compute_lqr_gain()
        
        # Состояние контроллера
        self.state = np.zeros((4, 1))  # [e, e_dot, psi, psi_dot]
        self.prev_lateral = 0
        self.prev_heading = 0
        
        print(f"LQR контроллер инициализирован")
        print(f"Усиление K: {self.K.flatten()}")
        
    def _compute_lqr_gain(self):
        # дискретное алгебраическое уравнение Риккати
        try:
            P = solve_discrete_are(self.A, self.B, self.Q, self.R)
            # матрица усиления
            K = np.linalg.inv(self.B.T @ P @ self.B + self.R) @ (self.B.T @ P @ self.A)
            return K
        except Exception as e:
            print(f"Ошибка вычисления LQR: {e}")
            # эмпирические коэффициенты
            return np.array([[-2.0, -1.0, -3.0, -0.5]])
    
    def update(self, lateral_error, heading_error):
        """
        Обновление LQR контроллера
        
        Args:
            lateral_error: латеральное отклонение (м)
            heading_error: ошибка угла курса (рад)
        
        Returns:
            control: управляющее воздействие (угол поворота колес)
        """
        # Вычисляем производные
        lateral_dot = (lateral_error - self.prev_lateral) / DT if self.prev_lateral != 0 else 0
        heading_dot = (heading_error - self.prev_heading) / DT if self.prev_heading != 0 else 0
        
        # Обновляем состояние
        self.state = np.array([
            [lateral_error],
            [lateral_dot],
            [heading_error],
            [heading_dot]
        ])
        
        # Вычисляем управление
        control = -self.K @ self.state
        
        # Сохраняем предыдущие значения
        self.prev_lateral = lateral_error
        self.prev_heading = heading_error
        
        # Ограничиваем управление
        control = np.clip(control, -1.5, 1.5)
        
        return float(control)
    
    def reset(self):
        """Сброс состояния контроллера"""
        self.state = np.zeros((4, 1))
        self.prev_lateral = 0
        self.prev_heading = 0
        
class LQRController:
    def __init__(self, max_linear_velocity=0.3, max_angular_velocity=8.0):
        self.max_linear_v = max_linear_velocity
        self.max_angular_v = max_angular_velocity

        # ручные коэффициенты LQR (аналог K)
        self.K = np.array([2.5, 2.0])   # [horizontal_error, angular_error]

    def update(self, horizontal_error, angular_error):
        x = np.array([horizontal_error, angular_error])
        omega = float(self.K @ x)

        omega = max(-self.max_angular_v, min(self.max_angular_v, omega))
        v = self.max_linear_v
        return v, omega


class MPCController:
    def __init__(self, horizon=15, dt=0.1,
                 max_linear_velocity=0.3,
                 max_angular_velocity=8.0):

        self.N = horizon
        self.dt = dt
        self.max_linear_v = max_linear_velocity
        self.max_angular_v = max_angular_velocity

        self.candidates = np.linspace(-max_angular_velocity,
                                      max_angular_velocity, 31)

    def simulate_cost(self, h0, a0, omega):
        h = h0
        a = a0
        cost = 0.0

        for _ in range(self.N):
            a = a - omega * self.dt
            h = h + a * self.dt

            #cost += 4*h*h + 2*a*a + 0.05*omega*omega
            cost += 4*h*h + 2*a*a + 0.3*omega*omega

        return cost

    def update(self, horizontal_error, angular_error):
        best_cost = 1e9
        best_u = 0.0

        for omega in self.candidates:
            c = self.simulate_cost(horizontal_error,
                                   angular_error,
                                   omega)
            if c < best_cost:
                best_cost = c
                best_u = omega

        return self.max_linear_v, float(best_u)
        
from gym_duckietown.envs import DuckietownEnv

lqr_controller = LQRController()
mpc_controller = MPCController()

env3 = DuckietownEnv(
	**{"seed": 128546,
	"map_name": "zigzag_dists", # где-то в репозитории можно эти карты настраивать
	"max_steps": 1000,
	"camera_width": 640,
	"camera_height": 480,
	"accept_start_angle_deg": 60, #what
	"full_transparency": True,
	"distortion": True,
	"domain_rand": False
	}
)

env2 = DuckietownEnv(
    **{"seed": 128546,
    "map_name": "straight_road", 
    "max_steps": 1000,
    "camera_width": 640,
    "camera_height": 480,
    "accept_start_angle_deg": 30, #what
    "full_transparency": True,
    "distortion": True,
    "domain_rand": False
    }
)



def run_simulation(controller_mode, pid_controller, lqr_controller, mpc_controller, algo_type,
                   max_steps=1000, show_display=True):
    done = False
    env2 = DuckietownEnv(
    **{"seed": 128546,
    "map_name": "loop_empty", 
    "max_steps": max_steps,
    "camera_width": 640,
    "camera_height": 480,
    "accept_start_angle_deg": 40, #what
    "full_transparency": True,
    "distortion": True,
    "domain_rand": False
    }
    )
    env = DuckietownEnv(
        **{"seed": 128546,
        "map_name": "straight_road",
        "max_steps": max_steps,
        "camera_width": 640,
        "camera_height": 480,
        "accept_start_angle_deg": 40,
        "full_transparency": True,
        "distortion": True,
        "domain_rand": False,
        "camera_rand": False,
        "user_tile_start": [0,0]
        }
    )
    obs = env.reset()
    step = 0
    controller_mode = controller_mode
    lateral_data = []
    heading_data = []
    horizontal_errors = []
    angular_errors = []
    omega_data = []
    height, width, _ = obs.shape
    if (algo_type == "lines"):
        yellow_processor = ImageLineProcessing(width, height, ColorLine.yellow)
        white_processor  = ImageLineProcessing(width, height, ColorLine.white)
    elif (algo_type == "con"):
        yellow_processor  = ContourLineProcessing(width, height, ColorLine.yellow)
        white_processor  = ContourLineProcessing(width, height, ColorLine.white)
    elif (algo_type == "haf"):
        yellow_processor  = HoughLineProcessing(width, height, ColorLine.yellow)
        white_processor  = HoughLineProcessing(width, height, ColorLine.white)
    elif (algo_type == "win"):
        yellow_processor  = SlidingWindowLineProcessing(width, height, ColorLine.yellow)
        white_processor  = SlidingWindowLineProcessing(width, height, ColorLine.white)
    
    process_times = []
    
    he_prev = 0.0
    center_x = 345
    lane_center_prev = center_x
    ang_prev = 0.0

    alpha = 0.2          # сглаживание
    delta_px = 20        # максимум сдвига центра за кадр (тюнить!)

    # --- loop ---
    while not done:
        original_img = obs

        tp = time.perf_counter()
        yellow_model = yellow_processor.process(original_img)
        white_model  = white_processor.process(original_img)
        process_time = (time.perf_counter() - tp) * 1000.0
        process_times.append(process_time)

        bottom_y = int(height * 0.78)

        valid = False   
        
        lane_width_px = 560

        # --- вычисление lane_center и угла ---
        if yellow_model["valid"] and white_model["valid"]:
            yx = yellow_processor.x_at_y(yellow_model, bottom_y)
            wx = white_processor.x_at_y(white_model, bottom_y)
            lane_center = (yx + wx) / 2

            ang = (math.atan(yellow_model["k"]) +
                   math.atan(white_model["k"])) / 2

            valid = True

        elif yellow_model["valid"]:
            yx = yellow_processor.x_at_y(yellow_model, bottom_y)

            # смещение внутрь полосы
            lane_center = yx + 100
            ang = math.atan(yellow_model["k"])

            valid = True

        elif white_model["valid"]:
            wx = white_processor.x_at_y(white_model, bottom_y)

            lane_center = wx - 100
            ang = math.atan(white_model["k"])

            valid = True

        else:
            # ❗ ничего не видим → держим прошлое
            lane_center = lane_center_prev
            ang = ang_prev

        # --- 1. УБИРАЕМ ВЫБРОСЫ (clip) ---
        lane_center = np.clip(
            lane_center,
            lane_center_prev - delta_px,
            lane_center_prev + delta_px
        )

        # --- 2. EMA сглаживание центра ---
        lane_center = alpha * lane_center + (1 - alpha) * lane_center_prev

        # --- 3. EMA сглаживание угла ---
        ang = alpha * ang + (1 - alpha) * ang_prev

        # --- обновляем состояния ---
        lane_center_prev = lane_center
        ang_prev = ang

        # --- 4. считаем ошибку ПОСЛЕ фильтрации ---
        horizontal_error = (lane_center - center_x) / width

        angular_error = ang
        pos = env.cur_pos
        angle = env.cur_angle
        lane = env.get_lane_pos2(pos, angle)
        lateral = lane.dist
        heading = lane.angle_rad
        
        if not valid:
            v, omega = 0.0, 0.0
            control_time = -1
        else:
            t0 = time.perf_counter()

            if controller_mode == "pid":
                #v, omega = pid_controller.update(horizontal_error, angular_error)
                v, omega = pid_controller.update(lateral, heading)

            elif controller_mode == "lqr":
                #v, omega = lqr_controller.update(horizontal_error, angular_error)
                v, omega = lqr_controller.update(lateral, heading)

            elif controller_mode == "mpc":
                #v, omega = mpc_controller.update(horizontal_error, angular_error)
                v, omega = mpc_controller.update(lateral, heading)

            control_time = (time.perf_counter() - t0) * 1000.0

        if show_display:
            dbg = yellow_processor.draw(original_img, yellow_model, (0,255,255))
            dbg = white_processor.draw(dbg, white_model, (255,255,255), int(lane_center), int(center_x))
            text_y = 40
            step_y = 20
            cv2.putText(dbg, f"omega {omega:.2f}", (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            text_y = text_y + step_y
            cv2.putText(dbg, f"he {horizontal_error:.2f} ae {angular_error:.2f}", (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            text_y = text_y + step_y
            cv2.putText(dbg, f"center_x {center_x:.2f} lane_center {lane_center:.2f}", (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            text_y = text_y + step_y
            cv2.putText(dbg, f"{controller_mode} {control_time:.2f} ms",
                (20, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,0),
                2)

            cv2.imshow("Processed (Debug)", dbg)
            cv2.waitKey(1)

        action = [v, omega]
        obs, rew, done, info = env.step(np.array(action))
        
        lateral_data.append(lateral)
        heading_data.append(heading)

        meters_per_pixel = 0.22 / lane_width_px
        horizontal_error_m = (lane_center - center_x) * meters_per_pixel
        horizontal_errors.append(horizontal_error_m)
        angular_errors.append(angular_error)
        omega_data.append(omega)
        #frame = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        #cv2.imshow("Duckietown (OpenCV)", original_img)
        #env.render()
    sred_time = sum(process_times)/len(process_times)
    print(f"Регулятор {controller_mode.upper()}: завершен")
    print(f"Среднее время {algo_type.upper()}: {sred_time}")
    with open('results.txt', 'a', encoding='utf-8') as f:
        f.write(f"Среднее время {algo_type.upper()}: {sred_time}\n")
    return lateral_data, heading_data, omega_data, horizontal_errors, angular_errors
def plot_comparison(all_lateral_data, controller_names, save_path='lateral_comparison.png'):
    """
    Построение графика сравнения lateral для всех регуляторов
    
    Parameters:
    - all_lateral_data: list of lists, данные lateral для каждого регулятора
    - controller_names: list of str, названия регуляторов
    - save_path: str, путь для сохранения графика
    """
    
    plt.figure(figsize=(14, 8))
    
    # Цвета для разных регуляторов
    colors = ['blue', 'red', 'green']
    linestyles = ['-', '--', '-.']
    
    for i, (lateral_data, name) in enumerate(zip(all_lateral_data, controller_names)):
        # Создаем массив времени (индексы шагов)
        time_steps = range(len(lateral_data))
        
        # Рисуем график
        plt.plot(time_steps, lateral_data, 
                color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)],
                linewidth=1.5,
                label=f'{name.upper()} Controller',
                alpha=0.8)
    
    # Настройка графика
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Lateral Distance (m)', fontsize=12)
    plt.title('Lateral Distance Comparison: PID vs LQR vs MPC', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='best', fontsize=10)
    
    # Добавляем горизонтальную линию на нуле (идеальное положение)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5, label='Ideal (0)')
    
    # Добавляем статистику на график
    stats_text = "Statistics (Mean ± Std):\n"
    for i, (lateral_data, name) in enumerate(zip(all_lateral_data, controller_names)):
        mean_val = np.mean(lateral_data)
        std_val = np.std(lateral_data)
        stats_text += f"{name.upper()}: {mean_val:.3f} ± {std_val:.3f}\n"
    
    # Добавляем текстовую информацию на график
    plt.text(0.02, 0.98, stats_text, 
             transform=plt.gca().transAxes,
             fontsize=9,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nГрафик сохранен как {save_path}")
    print(stats_text)


def plot_omega(all_heading_data, controller_names, save_path='omega_comparison.png'):
    """
    Построение графика сравнения heading для всех регуляторов
    """
    
    plt.figure(figsize=(14, 8))
    
    colors = ['blue', 'red', 'green']
    linestyles = ['-', '--', '-.']
    
    for i, (heading_data, name) in enumerate(zip(all_heading_data, controller_names)):
        time_steps = range(len(heading_data))
        plt.plot(time_steps, heading_data, 
                color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)],
                linewidth=1.5,
                label=f'{name.upper()} Controller',
                alpha=0.8)
    
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Angular speed (rad)', fontsize=12)
    plt.title('Angular speed Comparison: PID vs LQR vs MPC', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='best', fontsize=10)
    
    # Добавляем горизонтальные линии для справки
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    plt.axhline(y=math.pi/2, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Target (π/2)')
    
    # Добавляем статистику на график
    stats_text = "Statistics (Mean ± Std):\n"
    for i, (heading_data, name) in enumerate(zip(all_heading_data, controller_names)):
        mean_val = np.mean(heading_data)
        std_val = np.std(heading_data)
        stats_text += f"{name.upper()}: {mean_val:.3f} ± {std_val:.3f}\n"
    
    # Добавляем текстовую информацию на график
    plt.text(0.02, 0.98, stats_text, 
             transform=plt.gca().transAxes,
             fontsize=9,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
def plot_individual_heading(all_heading_data, controller_names, save_path='heading_comparison.png'):
    """
    Построение графика сравнения heading для всех регуляторов
    """
    
    plt.figure(figsize=(14, 8))
    
    colors = ['blue', 'red', 'green']
    linestyles = ['-', '--', '-.']
    
    for i, (heading_data, name) in enumerate(zip(all_heading_data, controller_names)):
        time_steps = range(len(heading_data))
        plt.plot(time_steps, heading_data, 
                color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)],
                linewidth=1.5,
                label=f'{name.upper()} Controller',
                alpha=0.8)
    
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Heading Angle (rad)', fontsize=12)
    plt.title('Heading Angle Comparison: PID vs LQR vs MPC', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='best', fontsize=10)
    
    # Добавляем горизонтальные линии для справки
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    plt.axhline(y=math.pi/2, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Target (π/2)')
    
    # Добавляем статистику на график
    stats_text = "Statistics (Mean ± Std):\n"
    for i, (heading_data, name) in enumerate(zip(all_heading_data, controller_names)):
        mean_val = np.mean(heading_data)
        std_val = np.std(heading_data)
        stats_text += f"{name.upper()}: {mean_val:.3f} ± {std_val:.3f}\n"
    
    # Добавляем текстовую информацию на график
    plt.text(0.02, 0.98, stats_text, 
             transform=plt.gca().transAxes,
             fontsize=9,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_error_simple(data1, data2):
    """
    Простой график ошибки между двумя массивами.
    
    Параметры:
    data1, data2 : array_like
        Два массива одинаковой длины
    """
    data1 = np.array(data1)
    data2 = np.array(data2)
    
    if len(data1) != len(data2):
        raise ValueError(f"Массивы имеют разную длину: {len(data1)} и {len(data2)}")
    
    error = data1 - data2
    
    plt.figure(figsize=(10, 6))
    
    # Основные данные
    plt.subplot(2, 1, 1)
    plt.plot(data1, 'b-', label='Data 1', linewidth=2)
    plt.plot(data2, 'r-', label='Data 2', linewidth=2)
    plt.title('Сравнение данных')
    plt.xlabel('Индекс')
    plt.ylabel('Значение')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Ошибка
    plt.subplot(2, 1, 2)
    plt.plot(error, 'g-', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.fill_between(range(len(error)), error, 0, alpha=0.3, color='green')
    plt.title(f'Ошибка (MSE = {np.mean(error**2):.4f})')
    plt.xlabel('Индекс')
    plt.ylabel('data1 - data2')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return error
    
cv2.destroyAllWindows()

# Список регуляторов для тестирования
controllers_to_test = ["lqr"]
#controllers_to_test = ["mpc"]
#controllers_to_test = ["pid"]
# Словарь для хранения данных всех запусков
all_lateral = []
all_heading = []
all_omega = []
all_hor_error = []
all_ang_error = []

pid_controller = PIDController(
        kp_horizontal = 0.9,
        ki_horizontal = 0.1,
        kd_horizontal = 0.22,
        kp_angular = 1.2,
        ki_angular = 0.4,
        kd_angular = 0.35,
        max_linear_velocity=0.5, #0.25
        max_angular_velocity=8 #1.5
    )
# Последовательный запуск для каждого регулятора
algo_types = ["lines"]  # или ["haf", "con", "lines", "win"]

for algo_type in algo_types:
    for idx, controller in enumerate(controllers_to_test):
        print(f"\n{'='*50}")
        print(f"Запуск {idx+1}/{len(controllers_to_test)}: Тестирование {controller.upper()} регулятора")
        print(f"Алгоритм: {algo_type.upper()}")
        print(f"{'='*50}")
        
        # Запуск симуляции
        lateral_data, heading_data, omega_data, herr_data, aerr_data = run_simulation(
            controller_mode=controller,
            pid_controller=pid_controller,
            lqr_controller=lqr_controller,
            mpc_controller=mpc_controller,
            algo_type=algo_type,  # перебираемый алгоритм
            max_steps=600,
            show_display=True
        )
        
        # Сохранение данных (можно добавить идентификатор алгоритма)
        all_lateral.append(lateral_data)
        all_heading.append(heading_data)
        all_omega.append(omega_data)
        all_hor_error.append(herr_data)
        all_ang_error.append(aerr_data)
        
        # Небольшая пауза между запусками
        time.sleep(1)
    
    # Пауза между сменой алгоритма (опционально)
    print(f"\nЗавершено тестирование всех регуляторов для алгоритма {algo_type.upper()}")
    time.sleep(1)

# Построение сравнительных графиков
print("\n" + "="*50)
print("Построение сравнительных графиков...")
print("="*50)

def plot_error_for_each_controller(all_heading, all_hor_error, controller_names):
    """
    Построение отдельных графиков для каждого регулятора
    """
    for idx, (heading, hor_error, name) in enumerate(zip(all_heading, all_hor_error, controller_names)):
        plt.figure(figsize=(10, 6))
        
        heading = np.array(heading).flatten()
        hor_error = np.array(hor_error).flatten()
        
        min_len = min(len(heading), len(hor_error))
        heading = heading[:min_len]
        hor_error = hor_error[:min_len]
        
        time_steps = range(min_len)
        
        plt.plot(time_steps, heading, 'b-', label='Heading', linewidth=2)
        plt.plot(time_steps, hor_error, 'r-', label='Horizontal Error', linewidth=2)
        
        error = heading - hor_error
        plt.fill_between(time_steps, error, 0, alpha=0.3, color='green')
        
        mae = np.mean(np.abs(error))
        mse = np.mean(error**2)
        
        plt.title(f'{name.upper()}: Heading vs Horizontal Error (MAE={mae:.4f})')
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
# График сравнения lateral
#
# График сравнения lateral
    
    # График сравнения heading
printNum = False
printPlot =  True
if printNum:
    print(all_lateral)
    print(all_heading)
if printPlot:
    plot_comparison(all_lateral, controllers_to_test, 'lateral_comparison.png')
    plot_individual_heading(all_heading, controllers_to_test, 'heading_comparison.png')
#plot_omega(all_omega, controllers_to_test, 'omega_comparison.png')
    plot_error_for_each_controller(all_heading, all_hor_error, controllers_to_test)
#plot_error_for_each_controller(all_lateral, all_ang_error, controllers_to_test)


# Дополнительно: сохранение всех данных в файл
#np.savez('all_controllers_data.npz',
#         pid_lateral=all_lateral[0],
#         pid_heading=all_heading[0],
#         lqr_lateral=all_lateral[1],
#         lqr_heading=all_heading[1],
#         mpc_lateral=all_lateral[2],
#         mpc_heading=all_heading[2])

#print("\nВсе данные сохранены в 'all_controllers_data.npz'")




