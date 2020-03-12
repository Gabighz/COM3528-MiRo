#!/usr/bin/env python

# This script makes MiRo wiggle its tail in a certain way

# Imports
##########################
import time
import os
import math
import numpy as np

import rospy  # ROS Python interface
from std_msgs.msg import Float32MultiArray, UInt32MultiArray, UInt16MultiArray, UInt8MultiArray, UInt16, Int16MultiArray, String
from sensor_msgs.msg import JointState, BatteryState, Imu, Range, CompressedImage
from geometry_msgs.msg import TwistStamped  # Velocity control messages

import miro2 as miro  # Import MiRo Developer Kit library


# Initialise a new ROS node to communicate with MiRo
rospy.init_node("wiggle_demo")
# Individual robot name acts as ROS topic prefix
topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")
# Create a new publisher to send commands to the robot
pub_cos = rospy.Publisher(topic_base_name + "/control/cosmetic_joints", Float32MultiArray, queue_size=0)

cos_joints = Float32MultiArray()
cos_joints.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

def wiggle(v, n, m):
    v = v + float(n) / float(m)
    if v > 2.0:
        v -= 2.0
    elif v > 1.0:
        v = 2.0 - v
    print "v = {}".format(v)
    return v

user_waggle = 0
wiggle_n = 15
droop = 0
wag =  1


MAX_TIME = 4*np.pi
def tailWag( t ):
    A = 0.5
    w = 2*np.pi
    f = lambda t: 0.5 + A*np.cos(w*t) # Wiggle function

    if t > MAX_TIME:
        cos_joints.data[wag] = 0.5
        r = False
    else:
        cos_joints.data[wag] = f(t)
        r = True

    cos_joints.data[droop] = 0.0
    print "wag = {}".format(f(t))

    return r

# Default wiggle function
def wiggle_tail():
    global user_waggle
    if user_waggle <= (2 * wiggle_n):
    	cos_joints.data[wag] = wiggle(0.5, user_waggle, wiggle_n)
    	cos_joints.data[droop] = 0.0
    else:
    	cos_joints.data[wag] = 0.5
    	cos_joints.data[droop] = wiggle(0.0, user_waggle - (2*wiggle_n), wiggle_n)


    user_waggle += 1
    if user_waggle > (4 * wiggle_n):
    	user_waggle = 0

t = 0.0

while not rospy.is_shutdown():
    # wiggle_tail()
    tailWag( t )
    t += 0.02
    assert np.abs(cos_joints.data[wag]) <= 1, "Wag values outside range"
    assert np.abs(cos_joints.data[droop]) <= 1, "Droop values outside range"
    pub_cos.publish(cos_joints)
    rospy.sleep(0.02)
