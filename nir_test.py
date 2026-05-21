import cv2
import numpy as np
import enum
import math
import os
import matplotlib.pyplot as plt

import psutil
import time

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
            rho=2,
            theta=np.pi / 90,
            threshold=50,
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
        self.debug = True
            
        if color == ColorLine.yellow:
            self.lower = np.array([20, 100, 100], np.uint8)
            self.upper = np.array([30, 255, 255], np.uint8)
            self.left, self.right = 0.0, 0.8
            self.debug_dir = "debug_yellow"


        else:
            self.lower = np.array([0,0,150], np.uint8)
            self.upper = np.array([180,100,255], np.uint8)
            self.left, self.right = 0.6, 1.0
            self.debug_dir = "debug_white"


        self.x1 = int(self.width * self.left)
        self.x2 = int(self.width * self.right)

        self.y1 = int(self.height * 0.35)
        self.y2 = int(self.height * 0.95)
        roi_h = self.y2 - self.y1

        self.scan_ys = np.linspace(
            0.05,
            roi_h - 1,
            11
        ).astype(int)
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

    import time
    import cv2
    import numpy as np
    import os

    def process(self, image, frame_id=None,  return_timing=False):
        if frame_id is not None and not self.debug:
            if frame_id > -1:
                frame_id = None
        t0 = time.perf_counter()

        if self.debug:
            os.makedirs(self.debug_dir, exist_ok=True)

        # ---------- ROI ----------
        roi = image[self.y1:self.y2, self.x1:self.x2]
        if roi.size == 0:
            print(f"EMPTY ROI: {self.y1}:{self.y2}, {self.x1}:{self.x2}")
        if self.debug:
            rgb_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{self.debug_dir}/{frame_id}_roi.png", rgb_roi)
        #rgb_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)

        #cv2.imwrite(f"{self.debug_dir}/{roi.size}_roi.png", rgb_roi)
        #self.stepik += 1
        t1 = time.perf_counter()

        # ---------- MASK ----------
        if self.color == ColorLine.yellow:
            mask = cv2.inRange(
                roi,
                np.array([140, 120, 0], dtype=np.uint8),
                np.array([255, 255, 160], dtype=np.uint8)
            )
        else:
            mask = cv2.inRange(
                roi,
                np.array([170, 170, 170], dtype=np.uint8),
                np.array([255, 255, 255], dtype=np.uint8)
            )

        if self.debug:
            cv2.imwrite(f"{self.debug_dir}/{frame_id}_mask.png", mask)

        t3 = time.perf_counter()

        # ---------- LINE DETECTION ----------
        result = self.detect_line_fast(mask, roi, frame_id)

        #t4 = time.perf_counter()

        # ---------- TIMING ----------
        #total = (t4 - t0) * 1000
        #roi_t = (t1 - t0) * 1000
        #mask_t = (t3 - t1) * 1000
        #detect_t = (t4 - t3) * 1000

        line, k, b, scan_time, fit_time = result
        if result is None:
            output = {"valid": False, "line": None, "k": None, "b": None}
        else:
            output = {"valid": True, "line": line, "k": k, "b": b}

        if return_timing:
            output["timing"] = {
                "roi": scan_time,
                "mask": mask_t,
                "detect": fit_time,
                "total": total,
                "area": roi.shape[0] * roi.shape[1]
            }

        return output
        
    def profile_roi_scaling(self, image, alphas=np.linspace(0.2, 1.0, 10), n_runs=30):
        import numpy as np

        results = {
            "area": [],
            "roi": [],
            "mask": [],
            "detect": []
        }

        # Сохраняем исходные параметры
        original_left = self.left
        original_right = self.right
        original_y1 = self.y1
        original_y2 = self.y2

        # Базовая "опорная точка" (слева снизу)
        base_left = 0.05
        base_bottom = 0.95

        # Диапазоны изменения ROI
        min_width, max_width = 0.2, 0.6
        min_height, max_height = 0.2, 0.5

        for a in alphas:
            # масштабируем ширину и высоту
            width = min_width + a * (max_width - min_width)
            height = min_height + a * (max_height - min_height)

            # ---------- ROI геометрия ----------
            self.left = base_left
            self.right = min(base_left + width, 1.0)

            self.y2 = int(self.height * base_bottom)
            self.y1 = int(self.height * (base_bottom - height))
            self.y1 = max(self.y1, 0)

            # пересчёт пиксельных координат
            self.x1 = int(self.width * self.left)
            self.x2 = int(self.width * self.right)

            
            roi_h = self.y2 - self.y1
            self.scan_ys = np.linspace(0, roi_h - 1, 11).astype(int)

            # ---------- сбор статистики ----------
            roi_times = []
            mask_times = []
            detect_times = []
            areas = []
            self.stepik = 1
            for _ in range(n_runs):
                res = self.process(image, return_timing=True)
                t = res["timing"]

                roi_times.append(t["roi"])
                mask_times.append(t["mask"])
                detect_times.append(t["detect"])
                areas.append(t["area"])

            results["area"].append(np.mean(areas))
            results["roi"].append(np.mean(roi_times))
            results["mask"].append(np.mean(mask_times))
            results["detect"].append(np.mean(detect_times))

        # ---------- вернуть всё обратно ----------
        self.left = original_left
        self.right = original_right
        self.y1 = original_y1
        self.y2 = original_y2

        self.x1 = int(self.width * self.left)
        self.x2 = int(self.width * self.right)

        roi_h = self.y2 - self.y1
        self.scan_ys = np.linspace(0, roi_h - 1, 11).astype(int)

        return results
        
    def plot_roi_scaling(self, results):
        import matplotlib.pyplot as plt

        # ===== ОСНОВНОЙ ГРАФИК =====
        plt.figure(figsize=(8, 5))

        plt.plot(results["area"], results["roi"],
                 linestyle='-', marker='o', label="Скан по срезам")

        plt.plot(results["area"], results["mask"],
                 linestyle='--', marker='s', label="Маска по цвету")

        plt.plot(results["area"], results["detect"],
                 linestyle=':', marker='^', label="Аппроксимация")

        plt.xlabel("Площадь ROI (пиксели)")
        plt.ylabel("Время (мс)")
        plt.title("Зависимость времени выполнения от размера ROI")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.savefig("roi_scaling_main.png", dpi=300)
        plt.show()

        # ===== ОТДЕЛЬНЫЙ ГРАФИК ДЛЯ DETECT =====
        plt.figure(figsize=(8, 5))

        plt.plot(results["area"], results["detect"],
                 linestyle='-', marker='o', label="Аппроксимация")

        plt.xlabel("Площадь ROI (пиксели)")
        plt.ylabel("Время (мс)")
        plt.title("Зависимость времени этапа детектирования от размера ROI")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.savefig("roi_scaling_detect.png", dpi=300)
        plt.show()

    def detect_line_fast(self, binary, roi, frame_id=None):
        t0 = time.perf_counter()

        h, w = binary.shape
        ys = self.scan_ys

        pts = []
        debug_img = None

        if self.debug and frame_id is not None:
            debug_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            for y in ys:
                if y >= h:
                    continue

                cv2.line(
                    debug_img,
                    (0, y),
                    (w - 1, y),
                    (250, 250, 2),
                    1
                )

        for y in ys:
            row = binary[y]
            xs = cv2.findNonZero(row.reshape(1, -1))

            if xs is None or len(xs) < 5:
                continue

            x = int(np.mean(xs[:, 0, 0]))
            pts.append((x, y))

            if self.debug:
                cv2.circle(debug_img, (x, y), 2, (0, 255, 255), -1)

        t1 = time.perf_counter()
        
        if len(pts) < 2:
            if self.debug and frame_id is not None:
                cv2.imwrite(f"{self.debug_dir}/{frame_id}_scan.png", debug_img)

            #print(f"SCAN:{(t1-t0)*1000:.5f} ms | NO LINE")
            return None

        pts = np.array(pts, dtype=np.float32)

        vx, vy, x0, y0 = cv2.fitLine(
            pts,
            cv2.DIST_L2,
            0,
            0.01,
            0.01
        )

        vx, vy, x0, y0 = vx[0], vy[0], x0[0], y0[0]

        if abs(vy) < 1e-5:
            return None

        y1 = int(h * 0.99)
        y2 = int(h * 0.01)

        x1 = int((y1 - y0) * vx / vy + x0)
        x2 = int((y2 - y0) * vx / vy + x0)


        k = float(vx / vy)
        b = float(x0 - k * y0)
        #if self.color == ColorLine.yellow:
            #print('yellow')
        #elif self.color == ColorLine.white:
        #    print('white')
        #print("vx ", vx, " vy ", vy, " x0 ", x0, " y0 ", y0, " k ", k, " b ", b)

        t2 = time.perf_counter()

        # ---------- DEBUG VIS ----------
        if self.debug:
            # точки
            for (x, y) in pts:
                cv2.circle(debug_img, (int(x), int(y)), 6, (143, 67, 250), -1)
            for i in range(1, len(pts)):
                x_prev, y_prev = pts[i-1]
                x_curr, y_curr = pts[i]
                cv2.line(debug_img,
                         (int(x_prev), int(y_prev)),
                         (int(x_curr), int(y_curr)),
                         (0, 200, 0), 1)
                        
            if frame_id is not None:
                cv2.imwrite(f"{self.debug_dir}/{frame_id}_line.png", debug_img)
            # линия
            cv2.line(debug_img, (x1, y1), (x2, y2), (3, 200, 254), 2)

            if frame_id is not None:
                cv2.imwrite(f"{self.debug_dir}/{frame_id}_line_with_points.png", debug_img)

        #print(
        #    f"SCAN:{(t1-t0)*1000:.5f} ms | "
        #    f"FIT:{(t2-t1)*1000:.5f} ms"
        #)

        return (x1 + self.x1, y1 + self.y1,
                x2 + self.x1, y2 + self.y1), k, b + self.x1 - k * self.y1, (t1-t0)*1000, (t2-t1)*1000

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
            kp_horizontal=0.8,
            ki_horizontal=0.2,
            kd_horizontal=0.5,
            kp_angular=1.5,
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
        self.R = np.array([[0.1]])  # Меньше вес на управление = более агрессивный контроль
        
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
    cpu_proc=[]
    mems = []
    process = psutil.Process()
    while not done:
            original_img = obs #warp_bird_eye(obs)


            tp = time.perf_counter()
            cpu_start = process.cpu_percent(interval=None)
            yellow_model = yellow_processor.process(original_img, step)
            white_model  = white_processor.process(original_img, step)
            cpu_usage = process.cpu_percent(interval=None)
            process_time = (time.perf_counter() - tp) * 1000.0
            step += 1
            memory = process.memory_info().rss / 1024 / 1024
            process_times.append(process_time)
            cpu_proc.append(cpu_usage)
            mems.append(memory)
            center_x = 360 #width / 2
            bottom_y = int(height * 0.78)

            horizontal_error = 0.0
            angular_error = 0.0
            valid = False
            lane_center = -1


            yx = -1
            wx = -1
            if yellow_model["valid"] and white_model["valid"]:
                yx = yellow_processor.x_at_y(yellow_model, bottom_y)
                wx = white_processor.x_at_y(white_model, bottom_y)

                lane_center = (yx + wx) / 2
                #print('lane center ', lane_center, ' wx ', wx, ' yx ', yx)
                horizontal_error = center_x - lane_center

                ang = (math.atan(yellow_model["k"]) + math.atan(white_model["k"])) / 2
                angular_error = ang
                valid = True

            elif yellow_model["valid"]:
                yx = yellow_processor.x_at_y(yellow_model, bottom_y)
                horizontal_error = yx - (center_x - 100)
                #print('lane center ', lane_center, ' wx ', wx, ' yx ', yx)
                angular_error = math.atan(yellow_model["k"])
                valid = True

            elif white_model["valid"]:
                wx = white_processor.x_at_y(white_model, bottom_y)
                horizontal_error = wx - (center_x + 100)
                #print('lane center ', lane_center, ' wx ', wx, ' yx ', yx)
                angular_error = math.atan(white_model["k"])
                valid = True

            horizontal_error /= width

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
                    v, omega = lqr_controller.update(lateral, heading)

                elif controller_mode == "mpc":
                    v, omega = mpc_controller.update(lateral, heading)

                control_time = (time.perf_counter() - t0) * 1000.0
            if show_display:
                original_img = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
                dbg = yellow_processor.draw(original_img, yellow_model, (0,255,255))
                dbg = white_processor.draw(dbg, white_model, (255,255,255), int(lane_center), int(center_x))
                text_y = 40
                step_y = 20
                #cv2.putText(dbg, f"omega {omega:.2f}", (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                #text_y = text_y + step_y
                cv2.putText(dbg, f"step {step}", (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                text_y = text_y + step_y
                cv2.putText(dbg, f"horizontal_error {horizontal_error:.2f} ", (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                text_y = text_y + step_y
                cv2.putText(dbg, f"angular_error {angular_error:.2f}", (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                #text_y = text_y + step_y
                #cv2.putText(dbg, f"center_x {center_x:.2f} lane_center {lane_center:.2f}", (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                #text_y = text_y + step_y
                #ky = yellow_model["k"]
                #by = yellow_model["b"]
                #cv2.putText(dbg, f"yellow {ky:.2f}  {by:.2f}", (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                #kw = white_model["k"]
                #bw = white_model["b"]
                #text_y = text_y + step_y
                #cv2.putText(dbg, f"white {kw:.2f}  {bw:.2f}", (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                #text_y = text_y + step_y
                #cv2.putText(dbg, f"wx {wx:.2f} yx {yx:.2f}", (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                #text_y = text_y + step_y
                #cv2.putText(dbg, f"{controller_mode} {control_time:.2f} ms",
                #    (20, text_y),
                #    cv2.FONT_HERSHEY_SIMPLEX,
                #    0.7,
                #    (0,255,0),
                #    2)
                
                cv2.imshow("Processed (Debug)", dbg)
                cv2.waitKey(1)

            action = [v, omega]
            obs, rew, done, info = env.step(np.array(action))

            lateral_data.append(lateral)
            heading_data.append(heading)
            horizontal_errors.append(horizontal_error)
            angular_errors.append(angular_error)
            omega_data.append(omega)
            #frame = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
            #cv2.imshow("Duckietown (OpenCV)", original_img)
            #env.render()
    sred_time = sum(process_times)/len(process_times)
    sred_cpu = sum(cpu_proc)/len(cpu_proc)
    sred_mem = sum(mems)/len(mems)
    print(f"Регулятор {controller_mode.upper()}: завершен")
    print(f"Среднее время {algo_type.upper()}: {sred_time}")
    with open('results.txt', 'a', encoding='utf-8') as f:
        f.write(f"Среднее время {algo_type.upper()}: {sred_time}\n")
        f.write(f"CPU {algo_type.upper()}: {sred_cpu}\n")
        f.write(f"Mem MB {algo_type.upper()}: {sred_mem}\n")
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
#controllers_to_test = ["pid", "lqr", "mpc"]
#controllers_to_test = ["mpc"]
controllers_to_test = ["pid"]
# Словарь для хранения данных всех запусков
all_lateral = []
all_heading = []
all_omega = []
all_hor_error = []
all_ang_error = []

pid_controller = PIDController(
        kp_horizontal = 3.05,
        ki_horizontal = 0.8,
        kd_horizontal = 0.5,
        kp_angular = 6.5,
        ki_angular = 0.8,
        kd_angular = 0.5,
        max_linear_velocity=0.3, #0.25
        max_angular_velocity=8 #1.5
    )
# Последовательный запуск для каждого регулятора
#algo_types = ["haf", "con", "lines", "win"]
algo_types = ["lines"]

env = DuckietownEnv(
        **{"seed": 128546,
        "map_name": "straight_road",
        "max_steps": 1,
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
height, width, _ = obs.shape
processor = ImageLineProcessing(width, height, ColorLine.yellow)
processor.debug = True


#results = processor.profile_roi_scaling(obs, n_runs=50)
#processor.plot_roi_scaling(results)

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
            max_steps=2,
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
printPlot =  False
if printNum:
    print(all_lateral)
    print(all_heading)
if printPlot:
    plot_comparison(all_lateral, controllers_to_test, 'lateral_comparison.png')
    plot_individual_heading(all_heading, controllers_to_test, 'heading_comparison.png')
#plot_omega(all_omega, controllers_to_test, 'omega_comparison.png')
    plot_error_for_each_controller(all_lateral, all_hor_error, controllers_to_test)
    plot_error_for_each_controller(all_heading, all_ang_error, controllers_to_test)


# Дополнительно: сохранение всех данных в файл
#np.savez('all_controllers_data.npz',
#         pid_lateral=all_lateral[0],
#         pid_heading=all_heading[0],
#         lqr_lateral=all_lateral[1],
#         lqr_heading=all_heading[1],
#         mpc_lateral=all_lateral[2],
#         mpc_heading=all_heading[2])

#print("\nВсе данные сохранены в 'all_controllers_data.npz'")




