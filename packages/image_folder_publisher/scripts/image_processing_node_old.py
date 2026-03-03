#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import math

from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3
from cv_bridge import CvBridge

from nir import ImageLineProcessing, ColorLine, warp_bird_eye


class ImageProcessingNode:
    def __init__(self):
        rospy.init_node("image_processing_node")

        self.bridge = CvBridge()

        self.sub = rospy.Subscriber("/image_raw", Image, self.callback, queue_size=1)
        self.pub_debug = rospy.Publisher("/lane_debug", Image, queue_size=1)
        self.pub_state = rospy.Publisher("/lane_state", Vector3, queue_size=1)

        rospy.loginfo("Image processing node started")

    def callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        h, w, _ = img.shape

        bird = warp_bird_eye(img)

        center_x = w / 2
        bottom_y = int(h * 0.78)

        yellow = ImageLineProcessing(bird, w, h, ColorLine.yellow)
        white  = ImageLineProcessing(bird, w, h, ColorLine.white)

        y_model = yellow.process()
        w_model = white.process()

        horizontal_error = 0.0
        angular_error = 0.0
        valid = False

        if y_model["valid"] and w_model["valid"]:
            yx = yellow.x_at_y(y_model, bottom_y)
            wx = white.x_at_y(w_model, bottom_y)

            lane_center = (yx + wx) / 2
            horizontal_error = lane_center - center_x

            ang = (math.atan(y_model["k"]) + math.atan(w_model["k"])) / 2
            angular_error = ang

            valid = True

        elif y_model["valid"]:
            yx = yellow.x_at_y(y_model, bottom_y)
            horizontal_error = yx - (center_x - 100)
            angular_error = math.atan(y_model["k"])
            valid = True

        elif w_model["valid"]:
            wx = white.x_at_y(w_model, bottom_y)
            horizontal_error = wx - (center_x + 100)
            angular_error = math.atan(w_model["k"])
            valid = True

        state = Vector3()
        state.x = horizontal_error
        state.y = angular_error
        state.z = 1.0 if valid else 0.0
        self.pub_state.publish(state)

        # Debug image
        dbg = yellow.draw(bird.copy(), y_model, (0,255,255))
        dbg = white.draw(dbg, w_model, (255,255,255))

        dbg_msg = self.bridge.cv2_to_imgmsg(dbg, "bgr8")
        self.pub_debug.publish(dbg_msg)


if __name__ == "__main__":
    ImageProcessingNode()
    rospy.spin()
