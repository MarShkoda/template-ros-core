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
from nir import ColorLine, warp_bird_eye
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped

collage_store = {}  # {frame_id: {"orig": img, "yellow": dbg_y, "white": dbg_w}}

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

        self.y1 = int(self.height * 0.35)
        self.y2 = int(self.height * 0.95)

    def _get_roi_mask(self, target_shape=None):
        """Create ROI mask, optionally resizing to target shape"""
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
            if len(xs) < 10:
                continue
            x = int(np.median(xs))
            pts.append((x, y))
            if debug:
                cv2.circle(dbg, (x, y), 4, (0,255,255), -1)

        if len(pts) < 2:
            if debug:
                self.pub_debug.publish(self.bridge.cv2_to_imgmsg(dbg, "bgr8"))
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
            self.pub_debug.publish(self.bridge.cv2_to_imgmsg(dbg, "bgr8"))

        # ---- Сохраняем для коллажа ----
        """  if orig_image is not None and frame_id is not None and frame_id < 100:
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
                del collage_store[frame_id] """ 

        #return (x1, y1), (x2, y2)
        return (x1,y1,x2,y2), float(vx/vy), float(x0 - (vx/vy)*y0)

    def process(self, image, frame_id=None):
        roi_rgb = image[self.y1:self.y2, self.x1:self.x2]
        hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_BGR2HSV)
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

class ImageProcessingNode(DTROS):
    def __init__(self, node_name):
        #rospy.init_node("image_processing_node")
        super(ImageProcessingNode, self).__init__(node_name=node_name, node_type=NodeType.DEBUG)

        self.bridge = CvBridge()
        self.frame_id = 0

         # ==================================================
        # Common params
        # ==================================================
        self.controller_type = rospy.get_param("~controller_type", "pid")

        self.max_linear_velocity = rospy.get_param(
            "~max_linear_velocity", 0.2
        )
        self.max_angular_velocity = rospy.get_param(
            "~max_angular_velocity", 8.0
        )

        # ==================================================
        # PID params
        # ==================================================
        self.kp_horizontal = rospy.get_param("~kp_horizontal", 2.0)
        self.ki_horizontal = rospy.get_param("~ki_horizontal", 0.0)
        self.kd_horizontal = rospy.get_param("~kd_horizontal", 0.5)

        self.kp_angular = rospy.get_param("~kp_angular", 4.9)
        self.ki_angular = rospy.get_param("~ki_angular", 0.0)
        self.kd_angular = rospy.get_param("~kd_angular", 0.5)

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
        self.mpc_horizon = rospy.get_param("~mpc_horizon", 15)
        self.mpc_dt = rospy.get_param("~mpc_dt", 0.1)

        self.mpc_q_h = rospy.get_param("~mpc_q_h", 4.0)
        self.mpc_q_a = rospy.get_param("~mpc_q_a", 2.0)
        self.mpc_r = rospy.get_param("~mpc_r", 0.3)

        self.mpc_candidates = rospy.get_param(
            "~mpc_candidates", 31
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
        self.sub = rospy.Subscriber("/autobot05/camera_node/image/compressed", CompressedImage, self.callback, queue_size=1)
        self.pub_debug = rospy.Publisher("/lane_debug", Image, queue_size=1)
        self.pub_state = rospy.Publisher("/lane_state", Vector3, queue_size=1)
        self.pub = rospy.Publisher("~car_cmd", Twist2DStamped, queue_size=1)


        self.yellow = None
        self.white = None

        #self.pid = PIDController(
        #    kp_horizontal=2.0,
        #    ki_horizontal=0,
        #    kd_horizontal=0.5,
        #    kp_angular=4.9,
        #    ki_angular=0,
        #    kd_angular=0.5,
        #    max_linear_velocity=0.2, #0.25
        #    max_angular_velocity=8 #1.5
        #)
        self.process_times = []
        self.control_times = []
        #self.pid = PIDController(
        #    kp_horizontal=0.008,
        #    ki_horizontal=0.0001,
        #    kd_horizontal=0.0005,
        #    kp_angular=0.3,
        #    ki_angular=0.0001,
        #    kd_angular=0.05,
        #    max_linear_velocity=0.1, #0.25
        #    max_angular_velocity=1.0 #1.5
        #)

        rospy.loginfo("Image processing node started *_*@")

    def callback(self, msg):
        self.frame_id += 1
        if (self.frame_id % 2 != 0):
            return
        img = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        h, w, _ = img.shape

        #bird = img #warp_bird_eye(img)

        if self.yellow is None:
            self.yellow = ImageLineProcessingOptimized(w, h, ColorLine.yellow)
            self.white  = ImageLineProcessingOptimized(w, h, ColorLine.white)

        center_x = 335 #w / 2
        bottom_y = int(h * 0.78)
        tp = time.perf_counter()
        y_model = self.yellow.process(img)
        w_model = self.white.process(img)
        process_time = (time.perf_counter() - tp) * 1000.0
        self.process_times.append(process_time)
      
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
            self.pub.publish(msg)
        else:
            v, w = self.pid.update(horizontal_error, angular_error)
            msg.v = v
            msg.omega = w
            self.pub.publish(msg)
        control_time = (time.perf_counter() - tc) * 1000.0
        self.control_times.append(control_time)
        #rospy.loginfo("+++++++")
        #rospy.loginfo(f"he {horizontal_error:.2f} ae {angular_error:.2f}")    
        #rospy.loginfo(f"msg.omega {msg.omega:.2f} msg.v {msg.v:.2f}")


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
