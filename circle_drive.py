#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import math

from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Vector3
from cv_bridge import CvBridge
from nir import PIDController
from nir import ColorLine, warp_bird_eye
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped


class ImageLineProcessingOptimized:
    def __init__(self, width, height, color: ColorLine):
        self.width = width
        self.height = height
        self.color = color
        self.bridge = CvBridge()
        if color == ColorLine.yellow:
            self.lower = np.array( [25,140,100], np.uint8)
            self.upper = np.array( [45,255,255], np.uint8)
            self.left, self.right = 0.0, 0.8
            self.pub_debug = rospy.Publisher("/lane_debug_yellow", Image, queue_size=1)

        else:
            self.lower = np.array([0,0,150], np.uint8)
            self.upper = np.array([180,100,255], np.uint8)
            self.left, self.right = 0.2, 1.0
            self.pub_debug = rospy.Publisher("/lane_debug_white", Image, queue_size=1)


        self._roi_mask = None
        self.image = None

    def _get_roi_mask(self):
        if self._roi_mask is None:
            mask = np.zeros((self.height, self.width), dtype=np.uint8)
            vertices = np.array([[
                (int(self.width * self.left),  int(self.height * 2/3)),
                (int(self.width * self.left),  int(self.height * 0.95)),
                (int(self.width * self.right), int(self.height * 0.95)),
                (int(self.width * self.right), int(self.height * 2/3)),
            ]], np.int32)
            cv2.fillPoly(mask, vertices, 255)
            self._roi_mask = mask
        return self._roi_mask

    def region_selection(self, color_mask):
        return cv2.bitwise_and(color_mask, self._get_roi_mask())

    def detect_line(self, binary):

        debug = self.pub_debug.get_num_connections() > 0

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

            if len(xs) < 15:
                continue

            x = int(np.mean(xs))

            pts.append((x, y))

            if debug:
                cv2.circle(dbg, (x, y), 4, (0,255,255), -1)

        if len(pts) < 2:
            if debug:
                self.pub_debug.publish(self.bridge.cv2_to_imgmsg(dbg, "bgr8"))
            return None

        pts = np.array(pts, np.float32)

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

        y1 = int(self.height * 0.78)
        y2 = int(self.height * 0.55)

        x1 = int((y1 - y0) * vx / vy + x0)
        x2 = int((y2 - y0) * vx / vy + x0)

        if debug:

            cv2.line(dbg, (x1,y1), (x2,y2), (0,0,255), 3)

            for y in ys:
                cv2.line(dbg, (0,y), (self.width,y), (255,0,0), 1)

            angle = np.degrees(np.arctan2(vy, vx))

            cv2.putText(
                dbg,
                f"angle={angle:.1f}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,255,255),
                2
            )

            self.pub_debug.publish(self.bridge.cv2_to_imgmsg(dbg, "bgr8"))

        return (x1,y1,x2,y2), float(vx/vy), float(x0 - (vx/vy)*y0)

    def process(self, image):
        #self.image = cv2.blur(image, (3,3))
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(self.image, self.lower, self.upper)
        roi = self.region_selection(mask)
        _, binary = cv2.threshold(roi, 30, 255, cv2.THRESH_BINARY)
        
        #self.pub_debug.publish(self.bridge.cv2_to_imgmsg(binary, "mono8"))

        result = self.detect_line(binary)
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
        self.sub = rospy.Subscriber("/autobot06/camera_node/image/compressed", CompressedImage, self.callback, queue_size=1)
        self.pub_debug = rospy.Publisher("/lane_debug", Image, queue_size=1)
        self.pub_state = rospy.Publisher("/lane_state", Vector3, queue_size=1)
        self.pub = rospy.Publisher("~car_cmd", Twist2DStamped, queue_size=1)


        self.yellow = None
        self.white = None

        self.pid = PIDController(
            kp_horizontal=0.001,
            ki_horizontal=0,
            kd_horizontal=0,
            kp_angular=0.25,
            ki_angular=0,
            kd_angular=0,
            max_linear_velocity=0.1, #0.25
            max_angular_velocity=0.3 #1.5
        )

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
        if (self.frame_id % 3 != 0):
            return
        img = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        h, w, _ = img.shape

        #bird = img #warp_bird_eye(img)

        if self.yellow is None:
            self.yellow = ImageLineProcessingOptimized(w, h, ColorLine.yellow)
            self.white  = ImageLineProcessingOptimized(w, h, ColorLine.white)

        center_x = w / 2
        bottom_y = int(h * 0.78)

        y_model = self.yellow.process(img)
        w_model = self.white.process(img)

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

        

        if (horizontal_error < 10):
            horizontal_error = 0
        horizontal_error *= 0.05
        if (angular_error < 0.0349066):#2 degs
            horizontal_error = 0
        state = Vector3()
        state.x = horizontal_error
        state.y = angular_error
        state.z = 1.0 if valid else 0.0
        self.pub_state.publish(state)

        msg = Twist2DStamped()
        #msg.v = 0.01
        #msg.omega = 0.01
        if not valid:
            msg.v = 0.0
            msg.omega = 0.0
            self.pub.publish(msg)
        else:
            v, w = self.pid.update(horizontal_error, angular_error)
            msg.v = v
            msg.omega = -w

            #self.pub.publish(msg)
        #rospy.loginfo("+++++++")
        #rospy.loginfo(f"he {horizontal_error:.2f} ae {angular_error:.2f}")    
        #rospy.loginfo(f"msg.omega {msg.omega:.2f} msg.v {msg.v:.2f}")


        dbg = self.yellow.draw(img, y_model, (0,255,255))
        dbg = self.white.draw(dbg, w_model, (255,255,255), int(lane_center), int(center_x))
        text_y = 40
        step_y = 20
        cv2.putText(dbg, f"omega {msg.omega:.2f}", (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        text_y = text_y + step_y
        cv2.putText(dbg, f"he {horizontal_error:.2f} ae {angular_error:.2f}", (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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