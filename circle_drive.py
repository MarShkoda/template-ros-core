#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import math
import time

from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Vector3
from cv_bridge import CvBridge
from nir import PIDController
from controllers import MPCController, LQRController
from compute import HoughLineProcessing, SlidingWindowLineProcessing, ContourLineProcessing
from nir import ColorLine
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped


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
class ImageLineProcessingOptimized:
    def __init__(self, width, height, color: 'ColorLine', save_debug_dir="debug_photos"):
        self.width = width
        self.height = height
        self.color = color
        self.bridge = CvBridge()
        self.save_debug_dir = save_debug_dir
        # os.makedirs(self.save_debug_dir, exist_ok=True)

        if color == ColorLine.yellow:
            #self.lower = np.array([25,140,100], np.uint8)
            #self.upper = np.array([45,255,255], np.uint8)
            self.lower = np.array([20, 80, 80])
            self.upper = np.array([50, 255, 255])
            self.left, self.right = 0.0, 0.8
            self.pub_debug = rospy.Publisher("/yellow_lane_debug", Image, queue_size=1)
            self.pub_debug_mask = rospy.Publisher("/yellow_mask", Image, queue_size=1)
            self.pub_debug_roi = rospy.Publisher("/yellow_roi", Image, queue_size=1)


        else:
            #self.lower = np.array([0,0,150], np.uint8)
            #self.upper = np.array([180,100,255], np.uint8)
            self.lower = np.array([0, 0, 200], dtype=np.uint8)
            self.upper = np.array([180, 50, 255], dtype=np.uint8)
            self.left, self.right = 0.6, 1.0
            self.pub_debug = rospy.Publisher("/white_lane_debug", Image, queue_size=1)
            self.pub_debug_mask = rospy.Publisher("/white_mask", Image, queue_size=1)
            self.pub_debug_roi = rospy.Publisher("/white_roi", Image, queue_size=1)


        self._roi_mask = None
        self.image = None
        self.x1 = int(self.width * self.left)
        self.x2 = int(self.width * self.right)

        self.y1 = int(self.height * 0.45)
        self.y2 = int(self.height * 0.95)


        roi_h = self.y2 - self.y1

        self.scan_ys = np.linspace(
            int(roi_h * 0.45),
            int(roi_h * 0.95),
            11
        ).astype(int)

    def region_selection(self, color_mask):
        roi_mask = self._get_roi_mask(color_mask.shape)
        return cv2.bitwise_and(color_mask, roi_mask)

    def detect_line_fast(self, binary):
        t0 = time.perf_counter()

        h, w = binary.shape

        ys = self.scan_ys

        pts = []

        for y in ys:
            row = binary[y]

            xs = cv2.findNonZero(row.reshape(1, -1))

            if xs is None or len(xs) < 5:
                continue

            x = int(np.mean(xs[:, 0, 0]))
            pts.append((x, y))

        t1 = time.perf_counter()

        if len(pts) < 2:
            print(f"SCAN:{(t1-t0)*1000:.1f} ms | NO LINE")
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

        y1 = int(h * 0.85)
        y2 = int(h * 0.45)

        x1 = int((y1 - y0) * vx / vy + x0)
        x2 = int((y2 - y0) * vx / vy + x0)

        k = float(vx / vy)
        b = float(x0 - k * y0)

        t2 = time.perf_counter()

        #print(
        #    f"SCAN:{(t1-t0)*1000:.1f} ms | "
        #    f"FIT:{(t2-t1)*1000:.1f} ms"
        #)

        return (x1 + self.x1, y1 + self.y1,
                x2 + self.x1, y2 + self.y1), k, b

    def process(self, image, frame_id=None):
        t0 = time.perf_counter()

        # ---------- ROI ----------
        roi = image[self.y1:self.y2, self.x1:self.x2]

        t1 = time.perf_counter()

        # --------------------

        t2 = time.perf_counter()

        # ---------- MASK ----------
        if self.color == ColorLine.yellow:
            mask = cv2.inRange(roi,
            np.array([0, 120,140], dtype = np.uint8),
            np.array([140, 255, 255], dtype = np.uint8)
            )
        else:
            mask = cv2.inRange(roi,
            np.array([170, 170, 170], dtype = np.uint8),
            np.array([255, 255, 255], dtype = np.uint8)
            )


        t3 = time.perf_counter()

        # ---------- LINE DETECTION ----------
        result = self.detect_line_fast(mask)

        t4 = time.perf_counter()

        # ---------- TIMING ----------
        total = (t4 - t0) * 1000
        roi_t = (t1 - t0) * 1000
        hsv_t = (t2 - t1) * 1000
        mask_t = (t3 - t2) * 1000
        detect_t = (t4 - t3) * 1000

        #print(
        #    f"ROI:{roi_t:.1f} ms | "
        #    f"HSV:{hsv_t:.1f} ms | "
        #    f"MASK:{mask_t:.1f} ms | "
        #    f"LINE:{detect_t:.1f} ms | "
        #    f"TOTAL:{total:.1f} ms"
        #)

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

class ImageProcessingNode(DTROS):
    def __init__(self, node_name):
        #rospy.init_node("image_processing_node")
        super(ImageProcessingNode, self).__init__(node_name=node_name, node_type=NodeType.DEBUG)

        self.bridge = CvBridge()
        self.frame_id = 0

        self.process_type = rospy.get_param("/process_type", "lines")


        # ==================================================
        # Common params
        # ==================================================
        self.controller_type = rospy.get_param("/controller_type", "pid")

        self.controller_type = rospy.get_param("/controller_type", "pid")

        self.max_linear_velocity = rospy.get_param(
            "~max_linear_velocity", 0.2
        )
        self.max_angular_velocity = rospy.get_param(
            "~max_angular_velocity", 8.0
        )

        # ==================================================
        # PID params
        # ==================================================
        self.kp_horizontal = rospy.get_param("/kp_horizontal", 2.0)
        self.ki_horizontal = rospy.get_param("/ki_horizontal", 0.0)
        self.kd_horizontal = rospy.get_param("/kd_horizontal", 0.5)

        self.kp_angular = rospy.get_param("/kp_angular", 4.9)
        self.ki_angular = rospy.get_param("/ki_angular", 0.0)
        self.kd_angular = rospy.get_param("/kd_angular", 0.5)

        # ==================================================
        # LQR params
        # ==================================================
        self.lqr_k_horizontal = rospy.get_param(
            "~lqr_k_horizontal", 2.5
        )
        self.lqr_k_angular = rospy.get_param(
            "~lqr_k_angular", 2.0
        )

        # ==================================================
        # MPC params
        # ==================================================
        self.mpc_horizon = rospy.get_param("/mpc_horizon", 15)
        self.mpc_dt = rospy.get_param("/mpc_dt", 0.1)

        self.mpc_q_h = rospy.get_param("/mpc_q_h", 4.0)
        self.mpc_q_a = rospy.get_param("/mpc_q_a", 2.0)
        self.mpc_r = rospy.get_param("/mpc_r", 0.3)

        self.mpc_candidates = rospy.get_param(
            "/mpc_candidates", 31
        )

        # ==================================================
        # Controller selection
        # ==================================================
        if self.controller_type == "pid":
            self.regulator = PIDController(
                kp_horizontal=self.kp_horizontal,
                ki_horizontal=self.ki_horizontal,
                kd_horizontal=self.kd_horizontal,

                kp_angular=self.kp_angular,
                ki_angular=self.ki_angular,
                kd_angular=self.kd_angular,

                max_linear_velocity=self.max_linear_velocity,
                max_angular_velocity=self.max_angular_velocity
            )
            rospy.loginfo(
                f"kp_horizontal: {self.kp_horizontal}"
            )
            rospy.loginfo(
                f"ki_horizontal: {self.ki_horizontal}"
            )
            rospy.loginfo(
                f"kd_horizontal: {self.kd_horizontal}"
            )
            rospy.loginfo(
                f"kp_angular: {self.kp_angular}"
            )
            rospy.loginfo(
                f"ki_angular: {self.ki_angular}"
            )
            rospy.loginfo(
                f"kd_angular: {self.kd_angular}"
            )
        elif self.controller_type == "lqr":
            self.regulator = LQRController(
                max_linear_velocity=self.max_linear_velocity,
                max_angular_velocity=self.max_angular_velocity
            )

            self.regulator.K = np.array([
                self.lqr_k_horizontal,
                self.lqr_k_angular
            ])

        elif self.controller_type == "mpc":
            self.regulator = MPCController(
                horizon=self.mpc_horizon,
                dt=self.mpc_dt,
                max_linear_velocity=self.max_linear_velocity,
                max_angular_velocity=self.max_angular_velocity
            )

            self.regulator.candidates = np.linspace(
                -self.max_angular_velocity,
                self.max_angular_velocity,
                self.mpc_candidates
            )

            self.regulator.q_h = self.mpc_q_h
            self.regulator.q_a = self.mpc_q_a
            self.regulator.r = self.mpc_r

        else:
            raise ValueError(
                f"Unknown controller_type: {self.controller_type}"
            )

        rospy.loginfo(
            f"Loaded controller: {self.controller_type}"
        )


        self.yellow = None
        self.white = None
        width = 640
        height = 480
        # ==================================================
        # ImageProcess selection
        # ==================================================
        if self.process_type == "lines":
            self.yellow = ImageLineProcessingOptimized(width, height, ColorLine.yellow)
            self.white  = ImageLineProcessingOptimized(width, height, ColorLine.white)
            
        elif self.process_type == "haf":
            self.yellow = HoughLineProcessing(width, height, ColorLine.yellow)
            self.white  = HoughLineProcessing(width, height, ColorLine.white)

        elif self.process_type == "win":
            self.yellow = SlidingWindowLineProcessing(width, height, ColorLine.yellow)
            self.white  = SlidingWindowLineProcessing(width, height, ColorLine.white)

        elif self.process_type == "con":
            self.yellow = ContourLineProcessing(width, height, ColorLine.yellow)
            self.white  = ContourLineProcessing(width, height, ColorLine.white)

        else:
            raise ValueError(
                f"Unknown process_type: {self.process_type}"
            )

        rospy.loginfo(
            f"Loaded image process type: {self.process_type}"
        )
        
        self.sub = rospy.Subscriber("/autobot06/camera_node/image/compressed", CompressedImage, self.callback, queue_size=1)
        self.pub_debug = rospy.Publisher("/lane_debug", Image, queue_size=1)
        self.pub_state = rospy.Publisher("/lane_state", Vector3, queue_size=1)
        self.pub = rospy.Publisher("~car_cmd", Twist2DStamped, queue_size=1)


        self.drive = rospy.get_param("/drive", 0)
        
        self.process_times = []
        self.process_cpu = []
        self.control_times = []
        
        rospy.loginfo("Image processing node started!")

    def callback(self, msg):
        self.frame_id += 1
        if (self.frame_id % 2 != 0):
            return
        img = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        h, w, _ = img.shape


        # if self.yellow is None:
        #     self.yellow = ImageLineProcessingOptimized(w, h, ColorLine.yellow)
        #     self.white  = ImageLineProcessingOptimized(w, h, ColorLine.white)

        center_x = 335
        bottom_y = int(h * 0.78)
        tp = time.perf_counter()
        cpu_start = time.process_time()
        y_model = self.yellow.process(img)
        w_model = self.white.process(img)
        process_time = (time.perf_counter() - tp) * 1000.0
        cpu_used = time.process_time() - cpu_start  # Реальное CPU время
        cpu_usage_percent = (cpu_used / (time.perf_counter() - tp)) * 100
        self.process_times.append(process_time)
        self.process_cpu.append(cpu_usage_percent)

        horizontal_error = 0.0
        angular_error = 0.0
        valid = False
        lane_center = 0
        if y_model["valid"] and w_model["valid"]:
            yx = self.yellow.x_at_y(y_model, bottom_y)
            wx = self.white.x_at_y(w_model, bottom_y)

            lane_center = (yx + wx) / 2
            horizontal_error = lane_center - center_x

            ang = (math.atan(y_model["k"]) + math.atan(w_model["k"])) / 2
            angular_error = ang
            valid = True

        elif y_model["valid"]:
            yx = self.yellow.x_at_y(y_model, bottom_y)
            lane_center = yx
            horizontal_error = yx - (center_x - 100)
            angular_error = math.atan(y_model["k"])
            valid = True

        elif w_model["valid"]:
            wx = self.white.x_at_y(w_model, bottom_y)
            lane_center = wx
            horizontal_error = wx - (center_x + 100)
            angular_error = math.atan(w_model["k"])
            valid = True

        
        horizontal_error = horizontal_error/w
        if (abs(angular_error) < 0.0001):
            angular_error = 0
        if (abs(horizontal_error) < 0.006): 
            horizontal_error = 0
        state = Vector3()
        state.x = horizontal_error
        state.y = angular_error
        state.z = 1.0 if valid else 0.0
        self.pub_state.publish(state)

        msg = Twist2DStamped()

        tc = time.perf_counter()
        if not valid:
            msg.v = 0.0
            msg.omega = 0.0
            
        else:
            v, w = self.regulator.update(horizontal_error, angular_error)
            msg.v = v
            msg.omega = w

        if (self.drive):
            self.pub.publish(msg)    
        control_time = (time.perf_counter() - tc) * 1000.0
        self.control_times.append(control_time)
        #rospy.loginfo("+++++++")
        #rospy.loginfo(f"he {horizontal_error:.2f} ae {angular_error:.2f}")    
        #rospy.loginfo(f"msg.omega {msg.omega:.2f} msg.v {msg.v:.2f}")

        if (self.frame_id == 22 or self.frame_id == 21):
            m_process = sum(self.process_times)/len(self.process_times)
            m_control = sum(self.control_times)/len(self.control_times)
            m_cpu = sum(self.process_cpu)/len(self.process_cpu)

            rospy.loginfo(f"m_process {m_process:.2f} m_control {m_control:.2f} m_cpu {m_cpu:.2f}")  
        dbg = self.yellow.draw(img, y_model, (0,255,255))
        dbg = self.white.draw(dbg, w_model, (255,255,255), int(lane_center), int(center_x))
        text_y = 40
        step_y = 20
        cv2.putText(dbg, f"omega {msg.omega:.2f}", (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        text_y = text_y + step_y
        cv2.putText(dbg, f"he {horizontal_error:.2f} ae {angular_error:.2f}", (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        text_y = text_y + step_y
        cv2.putText(dbg, f"center_x {center_x:.2f} lane_center {lane_center:.2f}", (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        self.pub_debug.publish(self.bridge.cv2_to_imgmsg(dbg, "bgr8"))

    def on_shutdown(self):
        """Shutdown procedure.

        Publishes a zero velocity command at shutdown."""
        msg = Twist2DStamped()
        msg.v = 0.0
        msg.omega = 0.0
        self.pub.publish(msg)

        super(ImageProcessingNode, self).on_shutdown()


if __name__ == "__main__":
    ImageProcessingNode(node_name='image_processing_node')
    rospy.spin()
