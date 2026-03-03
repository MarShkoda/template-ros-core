#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
import cv2
import os


class FolderImagePublisher:
    def __init__(self):
        rospy.init_node("folder_image_publisher")

        self.folder = rospy.get_param("~folder", "/home/user/images")
        self.rate = rospy.get_param("~rate", 5)

        self.pub = rospy.Publisher("/image_raw", Image, queue_size=1)

        rospy.loginfo(f"Reading images from: {self.folder}")
        self.files = sorted([
            f for f in os.listdir(self.folder)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])

        if not self.files:
            rospy.logerr("No images found in folder")
            exit(1)

        self.idx = 0
        self.run()

    def run(self):
        r = rospy.Rate(self.rate)

        while not rospy.is_shutdown():
            path = os.path.join(self.folder, self.files[self.idx])
            img = cv2.imread(path)

            if img is None:
                rospy.logwarn(f"Failed to load {path}")
                continue

            msg = self.cv_to_imgmsg(img)
            self.pub.publish(msg)

            rospy.loginfo(f"Published {self.files[self.idx]}")

            self.idx = (self.idx + 1) % len(self.files)
            r.sleep()

    def cv_to_imgmsg(self, img):
        msg = Image()
        msg.height, msg.width, _ = img.shape
        msg.encoding = "bgr8"
        msg.step = msg.width * 3
        msg.data = img.tobytes()
        return msg


if __name__ == "__main__":
    FolderImagePublisher()
