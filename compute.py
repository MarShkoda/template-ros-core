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