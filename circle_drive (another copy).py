#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage

import cv2
from cv_bridge import CvBridge

class CameraReaderNode(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(CameraReaderNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        # static parameters
        self._vehicle_name = 'autobot06'
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        # bridge between OpenCV and ROS
        self._bridge = CvBridge()
        # create window
        self._window = "camera-reader"
        print(cv2.__version__)
        # construct subscriber
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)

    def callback(self, msg):
        # convert JPEG bytes to CV image
        rospy.loginfo("CameraReaderNode :3")

        image = self._bridge.compressed_imgmsg_to_cv2(msg)
        # display frame
        #cv2.imshow(self._window, image)
        #cv2.waitKey(1)

if __name__ == '__main__':
    # create the node
    node = CameraReaderNode(node_name='camera_reader_node')
    # keep spinning
    rospy.spin()

#####################

#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import math

from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Vector3
from cv_bridge import CvBridge

from nir import ColorLine, warp_bird_eye


class ImageLineProcessingOptimized:
    def __init__(self, width, height, color: ColorLine):
        self.width = width
        self.height = height
        self.color = color

        if color == ColorLine.yellow:
            self.lower = np.array([27,45,176], np.uint8)
            self.upper = np.array([175,254,254], np.uint8)
            self.left, self.right = 0.0, 0.8
        else:
            self.lower = np.array([160,160,160], np.uint8)
            self.upper = np.array([255,255,255], np.uint8)
            self.left, self.right = 0.2, 1.0

        self._roi_mask = None
        self.image = None

    def _get_roi_mask(self):
        if self._roi_mask is None:
            mask = np.zeros((self.height, self.width), dtype=np.uint8)
            vertices = np.array([[
                (int(self.width * self.left),  int(self.height * 0.8)),
                (int(self.width * self.left),  int(self.height * 0.1)),
                (int(self.width * self.right), int(self.height * 0.1)),
                (int(self.width * self.right), int(self.height * 0.8)),
            ]], np.int32)
            cv2.fillPoly(mask, vertices, 255)
            self._roi_mask = mask
        return self._roi_mask

    def region_selection(self, color_mask):
        return cv2.bitwise_and(color_mask, self._get_roi_mask())

    def detect_line(self, binary):
        edges = cv2.Canny(binary, 80, 160)

        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180,
            threshold=20, minLineLength=30, maxLineGap=10
        )
        if lines is None:
            return None

        pts = []
        for l in lines:
            x1,y1,x2,y2 = l[0]
            if abs(x2-x1) < 5 or abs((y2-y1)/(x2-x1+1e-5)) > 0.3:
                pts.append((x1,y1))
                pts.append((x2,y2))

        if len(pts) < 6:
            return None

        pts = np.array(pts, np.float32)
        vx, vy, x0, y0 = cv2.fitLine(
            pts, cv2.DIST_L2, 0, 0.01, 0.01
        )
        vx, vy, x0, y0 = vx[0], vy[0], x0[0], y0[0]

        y1 = int(self.height * 0.78)
        y2 = int(self.height * 0.55)
        x1 = int((y1 - y0) * vx / vy + x0)
        x2 = int((y2 - y0) * vx / vy + x0)

        return (x1,y1,x2,y2), float(vx/vy), float(x0 - (vx/vy)*y0)

    def process(self, image):
        self.image = cv2.blur(image, (3,3))

        mask = cv2.inRange(self.image, self.lower, self.upper)
        roi = self.region_selection(mask)
        _, binary = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)

        result = self.detect_line(binary)
        if result is None:
            return {"valid": False, "line": None, "k": None, "b": None}

        line, k, b = result
        return {"valid": True, "line": line, "k": k, "b": b}

    def x_at_y(self, model, y):
        if not model["valid"]:
            return None
        return model["k"] * y + model["b"]

    def draw(self, image, model, color=(0,255,0), thickness=2):
        if not model["valid"]:
            return image
        x1,y1,x2,y2 = model["line"]
        return cv2.line(image, (x1,y1), (x2,y2), color, thickness)


class ImageProcessingNode:
    def __init__(self):
        rospy.init_node("image_processing_node")

        self.bridge = CvBridge()

        self.sub = rospy.Subscriber("/camera_node/image/compressed", CompressedImage, self.callback, queue_size=1)
        self.pub_debug = rospy.Publisher("/lane_debug", Image, queue_size=1)
        self.pub_state = rospy.Publisher("/lane_state", Vector3, queue_size=1)

        self.yellow = None
        self.white = None

        rospy.loginfo("Image processing node started")

    def callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        h, w, _ = img.shape

        bird = warp_bird_eye(img)

        if self.yellow is None:
            self.yellow = ImageLineProcessingOptimized(w, h, ColorLine.yellow)
            self.white  = ImageLineProcessingOptimized(w, h, ColorLine.white)

        center_x = w / 2
        bottom_y = int(h * 0.78)

        y_model = self.yellow.process(bird)
        w_model = self.white.process(bird)

        horizontal_error = 0.0
        angular_error = 0.0
        valid = False

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
            horizontal_error = yx - (center_x - 100)
            angular_error = math.atan(y_model["k"])
            valid = True

        elif w_model["valid"]:
            wx = self.white.x_at_y(w_model, bottom_y)
            horizontal_error = wx - (center_x + 100)
            angular_error = math.atan(w_model["k"])
            valid = True

        state = Vector3()
        state.x = horizontal_error
        state.y = angular_error
        state.z = 1.0 if valid else 0.0
        self.pub_state.publish(state)

        dbg = self.yellow.draw(bird.copy(), y_model, (0,255,255))
        dbg = self.white.draw(dbg, w_model, (255,255,255))

        self.pub_debug.publish(self.bridge.cv2_to_imgmsg(dbg, "bgr8"))


if __name__ == "__main__":
    ImageProcessingNode()
    rospy.spin()