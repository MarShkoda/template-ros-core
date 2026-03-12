#!/usr/bin/env python3
import sys
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped

class MyNode(DTROS):

    def __init__(self, node_name):
        super(MyNode, self).__init__(node_name=node_name, node_type=NodeType.DEBUG)
        self.pub = rospy.Publisher("~car_cmd", Twist2DStamped, queue_size=1)

    def run(self):
        # publish message every 1 second
        rate = rospy.Rate(1) # 1Hz
        while not rospy.is_shutdown():
            msg = Twist2DStamped()
            msg.v = 0.5
            msg.omega = 1.0
            rospy.loginfo("Publishing message 0/0.5 :3")
            self.pub.publish(msg)
            rate.sleep()
            msg.omega = 0.0
            rospy.loginfo("Publishing message 0/0.0")
            self.pub.publish(msg)
            rate.sleep()
            sys.stdout.flush()
            
    def on_shutdown(self):
        """Shutdown procedure.

        Publishes a zero velocity command at shutdown."""
        msg = Twist2DStamped()
        msg.v = 0.0
        msg.omega = 0.0
        self.pub.publish(msg)

        super(MyNode, self).on_shutdown()

if __name__ == '__main__':
    # create the node
    node = MyNode(node_name='circle_drive_node')
    # run node
    node.run()
    # keep spinning
    rospy.spin()
