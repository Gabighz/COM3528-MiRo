#!/usr/bin/env python

# This script makes MiRo look for a particular ball and kick it when found

# Imports
##########################
import os
import sys
from math import radians  # This is used for the head pose
import numpy as np  # Numerical Analysis library
import cv2  # Computer Vision library

import rospy  # ROS Python interface
from sensor_msgs.msg import CompressedImage  # ROS CompressedImage messages
from sensor_msgs.msg import JointState  # Head joints state
from cv_bridge import CvBridge, CvBridgeError  # ROS -> OpenCV converter
from geometry_msgs.msg import TwistStamped  # Velocity control messages


import miro2 as miro  # Import MiRo Developer Kit library
##########################

class ClientBall:

    # Class constants
    TICK = 0.01 # This is the main control loop update interval in secs
    CAM_FREQ = 1 # Number of ticks before camera gets a new frame. Increase in case of network lag


    # Reset MiRo head to default position, to avoid having tilted frames
    def reset_head_pose(self):
        self.kin_joints = JointState()
        self.kin_joints.name = ["tilt", "lift", "yaw", "pitch"]
        self.kin_joints.position = [0.0, radians(34.0), 0.0, 0.0]
        t = 0
        while not rospy.core.is_shutdown():
            self.pub_kin.publish(self.kin_joints)
            rospy.sleep(self.TICK)
            t += self.TICK
            if t > 1:
                break


    # This function simplifies driving MiRo
    def drive(self, speed_l = 0.1, speed_r = 0.1):  # (m/sec, m/sec)
        # Prepare an empty velocity command message
        msg_cmd_vel = TwistStamped()

        # Desired wheel speed (m/sec)
        wheel_speed = [speed_l, speed_r]

        # Convert wheel speed to command velocity (m/sec, Rad/sec)
        (dr, dtheta) = miro.utils.wheel_speed2cmd_vel(wheel_speed)

        # Update message to publish to control/cmd_vel
        msg_cmd_vel.twist.linear.x = dr
        msg_cmd_vel.twist.angular.z = dtheta

        # Publish message to topic
        self.vel_pub.publish(msg_cmd_vel)


    # This callback function will be executed upon image arrival
    def callback_cam(self, ros_image, index):
        # Silently(-ish) handle corrupted JPEG frames
        try:
            # Convert compressed ROS image to raw CV image
            image = self.image_converter.compressed_imgmsg_to_cv2(ros_image, "rgb8")
            # Convert from OpenCV's default BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Store image as class attribute for further use
            self.input_camera[index] = image
            # Get image dimensions
            self.frame_height, self.frame_width, channels = image.shape
            self.x_centre = self.frame_width / 2.0
            self.y_centre = self.frame_height / 2.0
            # A new frame is available for processing
            self.new_frame[index] = True
        except CvBridgeError as e:
            # Ignore corrupted frames
            pass

    def callback_caml(self, ros_image): # Left camera
        self.callback_cam(ros_image, 0)

    def callback_camr(self, ros_image): # Right camera
        self.callback_cam(ros_image, 1)


    # This function performs image processing to find a ball in a given frame
    def detect_ball(self, frame, index):
        """
        This function is fine-tuned to detect a small, blue ball.
        Adapt the parameters to detect the ball your group is using.
        If that proves too be dificult, you can try detecting simple-coloured
        circles, printed out or displayed on a screen.

        Hint: Try changing the values on the following lines:
        109, 115-116, 136-138, 141-145
        """

        if frame is None: # Sanity check
            return

        # Flag this frame as processed, so that it's not reused in case of lag
        self.new_frame[index] = False
        # Get image in HSV (hue, saturation, value) colour format
        im_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        # Specify target ball colour
        rgb_colour = np.uint8([[[0, 0, 255]]])  # e.g. Blue
        # Convert this colour to HSV colour model
        hsv_colour = cv2.cvtColor(rgb_colour, cv2.COLOR_RGB2HSV)

        # Extract colour boundaries for masking image
        target_hue = hsv_colour[0, 0][0]  # Get the hue value from the numpy array containing target colour
        hsv_lo_end = np.array([target_hue - 20, 70, 70])
        hsv_hi_end = np.array([target_hue + 20, 255, 255])

        # # Alternative method to separate desired colour range
        # # Specify main ball colour range in RGB for colour segmentation
        # rgb_lo_end = np.uint8([[[5, 5, 5]]]) # e.g., a blue ball
        # rgb_hi_end = np.uint8([[[50, 50, 200]]])
        # # Convert the range to HSV
        # hsv_lo_end = cv2.cvtColor(rgb_lo_end, cv2.COLOR_RGB2HSV)
        # hsv_hi_end = cv2.cvtColor(rgb_hi_end, cv2.COLOR_RGB2HSV)

        # Generate the mask based on the desired hue range
        mask = cv2.inRange(im_hsv, hsv_lo_end, hsv_hi_end)

        # # Debug window (uncomment to enable)
        mask_on_image = cv2.bitwise_and(im_hsv, im_hsv, mask=mask)
        cv2.imshow('mask'+ str(index), mask_on_image)
        cv2.waitKey(1)

        # Clean up the image
        seg = mask
        seg = cv2.GaussianBlur(seg, (5, 5), 0)
        seg = cv2.erode(seg, None, iterations=2)
        seg = cv2.dilate(seg, None, iterations=2)

        # Fine-tune parameters
        ball_detect_min_dist_between_cens = 40 # Empirical
        canny_high_thresh = 10 # Empirical
        ball_detect_sensitivity = 20 # Lower detects more circles, so it's a trade-off
        ball_detect_min_radius = 5 # Arbitrary, empirical
        ball_detect_max_radius = 50 # Arbitrary, empirical

        # Find circles using OpenCV routine
        # This fucntion returns a list of circles', with their x, y and r values
        circles = cv2.HoughCircles(seg, cv2.HOUGH_GRADIENT,
                1, ball_detect_min_dist_between_cens, \
                param1=canny_high_thresh, param2=ball_detect_sensitivity, \
                minRadius=ball_detect_min_radius, maxRadius=ball_detect_max_radius)

        if circles is None:
            # If no circles were found, just display the original image
            cv2.imshow("camera" + str(index), frame)
            cv2.waitKey(1)
            return
        # Get the largest circle
        max_circle = None
        self.max_rad = 0
        circles = np.uint16(np.around(circles))
        for c in circles[0,:]:
            if c[2] > self.max_rad:
                self.max_rad = c[2]
                max_circle = c
        # This shouldn't happen, but you never know
        if max_circle is None:
            return

        # Append detected circle and its centre to the frame
        cv2.circle(frame,(max_circle[0],max_circle[1]),max_circle[2],(0,255,0),2)
        cv2.circle(frame,(max_circle[0],max_circle[1]),2,(0,0,255),3)
        cv2.imshow("camera" + str(index), frame)
        cv2.waitKey(1)

        # Normalise values to: x,y = [-0.5, 0.5], r = [0, 1]
        max_circle = np.array(max_circle).astype('float32')
        max_circle[0] -= self.x_centre
        max_circle[0] /= self.frame_width
        max_circle[1] -= self.y_centre
        max_circle[1] /= self.frame_width
        max_circle[1] *= -1.0
        max_circle[2] /= self.frame_width

        # Convert values to list
        return [max_circle[0], max_circle[1], max_circle[2]]


    # Move MiRO until it sees a ball
    def look_for_ball(self):
        """
        This function should somehow move MiRo if it doesn't see a ball in its
        current position, until it sees one.
        Hint: The 'lock_on_ball' function contains a method of checking for
        successful ball detection!

        Replace placeholder code 'rospy.sleep(1)' with your implementation
        """
        rospy.sleep(1)
        # Once a ball has been detected
        print("MiRo is locking in on the ball...")
        self.status_code = 2 # Selector to locking in


    # Once a ball has been detected, this function makes sure MiRo faces it
    def lock_on_ball(self, error=25):
        # For each camera (0 = left, 1 = right)
        for index in range(2):
            # Skip if there's no new image (in case network is choking)
            if not self.new_frame[index]:
                continue
            image = self.input_camera[index]
            # Run the detect ball procedure
            self.ball[index] = self.detect_ball(image, index)
        # If only the right camera sees the ball, rotate clockwise
        if not self.ball[0] and self.ball[1]:
            self.drive(0.015, -0.015)
        # Conversely, rotate counterclockwise
        elif self.ball[0] and not self.ball[1]:
            self.drive(-0.015, 0.015)
        # If we see the ball with both cameras, make the MiRo face the ball
        elif self.ball[0] and self.ball[1]:
            error = 0.05 # 5% of image width
            # Use the normalised values
            left_x = self.ball[0][0] # should be in range [0.0, 0.5]
            right_x = self.ball[1][0] # should be in range [-0.5, 0.0]
            rotation_speed = 0.01 #  Even slower
            if abs(left_x) - abs(right_x) > error:
                self.drive(rotation_speed, -rotation_speed) # turn clockwise
            elif abs(left_x) - abs(right_x) < -error:
                self.drive(-rotation_speed, rotation_speed) # turn counterclockwise
            else:
                # We've succesfully turned to face the ball
                self.status_code = 3
                print("MiRo is kicking the ball...")
        # Otherwise, we've lost the ball :-(
        else:
            self.status_code = 0
            print("MiRo has lost the ball...")


    # GOAAAL
    def kick(self):
        """
        Once MiRO is in position, this function should drive the MiRo forward
        until it kicks the ball!

        Replace placeholder code 'rospy.sleep(1)' with your implementation
        """
        rospy.sleep(1)
        print("Chaaarge!...")
        self.status_code = 0  # Selector to default position after the kick


    # This function is called upon class initialisation
    def __init__(self):
        # Initialise a new ROS node to communicate with MiRo
        rospy.init_node("lab6", anonymous=True)
        # Give it a second to make sure everything is initialised
        rospy.sleep(1.0)
        # Initialise CV Bridge
        self.image_converter = CvBridge()
        # Individual robot name acts as ROS topic prefix
        topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")
        # Create two new subscribers to recieve camera images with attached callbacks
        self.sub_caml = rospy.Subscriber(topic_base_name + "/sensors/caml/compressed",
            CompressedImage, self.callback_caml, queue_size=1, tcp_nodelay=True)
        self.sub_camr = rospy.Subscriber(topic_base_name + "/sensors/camr/compressed",
            CompressedImage, self.callback_camr, queue_size=1, tcp_nodelay=True)
        # Create a new publisher to send velocity commands to the robot
        self.vel_pub = rospy.Publisher(topic_base_name + "/control/cmd_vel",
            TwistStamped, queue_size=0)
        # Create a new publisher to move the robot head
        self.pub_kin = rospy.Publisher(topic_base_name + "/control/kinematic_joints",
            JointState, queue_size=0)
        # Create handle to store images
        self.input_camera = [None, None]
        # New frame notification
        self.new_frame = [False, False]
        # Create variable to store a list of ball's x, y, and r values for each camera
        self.ball = [None, None]
        # Set the default frame width
        self.frame_width = 640
        # Move the head to default pose
        self.reset_head_pose()

    # This is the main control loop
    def loop(self):
        print "MiRo plays ball, press CTRL+C to halt..."
        # Main control loop iteration counter
        self.counter = 0
        # This switch loops through MiRo behaviours:
        # Find ball, lock on the ball and kick ball
        self.status_code = 0
        while not rospy.core.is_shutdown():

            # Step 1. Find ball
            if self.status_code == 1:
                # Every once in a while, look fot ball
                if self.counter % self.CAM_FREQ == 0:
                    self.look_for_ball()

            # Step 2. Orient yourself
            elif self.status_code == 2:
                self.lock_on_ball()

            # Step 3. Kick!
            elif self.status_code == 3:
                self.kick()

            # Fall back
            else:
                self.status_code = 1
                print("MiRo is looking for the ball...")

            # Yield
            self.counter += 1
            rospy.sleep(self.TICK)


# This condition fires when we call the script directly
if __name__ == "__main__":
	main = ClientBall() # Instantiate class
	main.loop() # Run the main control loop
