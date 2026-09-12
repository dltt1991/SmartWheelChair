"""Gazebo world and odom coincide in the simulation worlds."""
from copy import deepcopy

from nav_msgs.msg import Odometry
import rospy


def odom_frame_message(message):
    result = deepcopy(message)
    result.header.frame_id = 'odom'
    return result


def main():
    rospy.init_node('odom_frame_relay')
    publisher = rospy.Publisher('/odom', Odometry, queue_size=1)
    subscriber = rospy.Subscriber(
        '/rear_axle_ground_truth', Odometry,
        lambda message: publisher.publish(odom_frame_message(message)), queue_size=1)
    rospy.spin()
