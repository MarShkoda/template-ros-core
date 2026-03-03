#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist, Vector3
from nir import PIDController


class ControllerNode:
    def __init__(self):
        rospy.init_node("pid_controller")

        self.pid = PIDController(
            kp_horizontal=0.008,
            ki_horizontal=0.0001,
            kd_horizontal=0.0005,
            kp_angular=0.3,
            ki_angular=0.0001,
            kd_angular=0.05,
            max_linear_velocity=0.25,
            max_angular_velocity=1.5
        )

        self.sub = rospy.Subscriber("/lane_state", Vector3, self.callback, queue_size=1)
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        rospy.loginfo("PID controller started")

    def callback(self, msg):
        horizontal_error = msg.x
        angular_error = msg.y
        valid = msg.z > 0.5

        if not valid:
            self.publish(0.0, 0.0)
            return

        v, w = self.pid.update(horizontal_error, angular_error)
        self.publish(v, w)

    def publish(self, v, w):
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        #rospy.loginfo(f"PID controller: v={v:.2f}, w={w:.2f}")
        self.pub.publish(cmd)


if __name__ == "__main__":
    ControllerNode()
    rospy.spin()
