#!/usr/bin/python

import rospy
import miro2 as miro
import os

from std_msgs.msg import Float32MultiArray, UInt32MultiArray
from geometry_msgs.msg import TwistStamped

import numpy as np
import time


# Constants
MAX_TIME = 10.0
# Utility enums
tilt, lift, yaw, pitch = range(4)
droop, wag, left_eye, right_eye, left_ear, right_ear = range(6)

class MiroController:

	def __init__( self ):
		print "Initializing the controller"
		self.actions = [ self.earWiggle, self.tailWag, self.rotate, self.shine ]
		self.Q = [0]*4
		self.N = [0]*4
		self.r = 0

		# Set robot name
		topic_root = "/" + os.getenv("MIRO_ROBOT_NAME")
		# Python needs to initialise a ROS node for publishing data
		rospy.init_node("kbandit", anonymous=True)
		# Define ROS publishers
		self.pub_cmd_vel = rospy.Publisher(topic_root + "/control/cmd_vel", TwistStamped, queue_size=0)
		self.pub_cos = rospy.Publisher(topic_root + "/control/cosmetic_joints", Float32MultiArray, queue_size=0)
		self.pub_illum = rospy.Publisher(topic_root + "/control/illum", UInt32MultiArray, queue_size=0)

		# Subscribers
		rospy.Subscriber(topic_root + '/sensors/package', miro.msg.sensors_package, self.touchListener)

		# Initializing object for data publishing
		self.velocity = TwistStamped()
		self.cos_joints = Float32MultiArray()
		self.cos_joints.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
		self.illum = UInt32MultiArray()
		self.illum.data = [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF]


	# Actions
	def earWiggle( self, t ):
		A = 1.0
		w = 2*np.pi*0.2
		f = lambda t: A*np.cos(w*t)

		if t > MAX_TIME:
			self.cos_joints.data[left_ear] = 0.0
			self.cos_joints.data[right_ear] = 0.0
			r = False
		else:
			self.cos_joints.data[left_ear] = f(t)
			self.cos_joints.data[right_ear] = f(t)
			r = True

		
		self.pub_cos.publish(self.cos_joints)
		#print "wiggle"

		return r
		

	def tailWag( self, t ):
		A = 1.0
		w = 2*np.pi*0.2
		f = lambda t: A*np.cos(w*t)

		if t > MAX_TIME:
			self.cos_joints.data[wag] = 0.5
			r = False
		else:
			self.cos_joints.data[wag] = f(t)
			r = True

		self.cos_joints.data[droop] = 0.0
		self.pub_cos.publish(self.cos_joints)
		#print "wag"

		return r

		
	def rotate( self, t ):

		if t > MAX_TIME:
			l_val = 0.0
			r_val = 0.0
			r = False
		else:
			l_val = 1.0
			r_val = 0.0	
			r = True

		wheel_speed = [l_val, r_val]
		(dr, dtheta) = miro.utils.wheel_speed2cmd_vel(wheel_speed)
		self.velocity.twist.linear.x = dr
		self.velocity.twist.angular.z = dtheta		
		self.pub_cmd_vel.publish(self.velocity)
		#print "rotate"

		return r


	def shine( self, t ):
		color = 0xFF00FF00
		f = lambda t: t*30/MAX_TIME
		ip = int(np.mod(np.rint(f(t)), 6))
		i = int(np.mod(ip + 1, 6))
		self.illum.data[ip] = 0x00000000

		if t > MAX_TIME:
			self.illum.data[i] = 0x00000000			
			r = False
		else:
			self.illum.data[i] = color			
			r = True

		self.pub_illum.publish(self.illum)
		#print "Shinnning"

		return r

	# Listeners
	def touchListener( self, data ):
		touch_data = data.touch_body.data
		strength = 0

		for i in range(0, 14):
			if ((1 << i) & touch_data) > 0:
				strength += 1

		if strength > 2:
			reward = 1.0
		else:
			reward = 0.0

		print touch_data

		self.Q[self.r] += (reward - self.Q[self.r])/float(self.N[self.r])

	# Main loop
	def run( self ):
		running = True
		self.r = np.random.randint( 0, len(self.actions) )
		self.N[self.r] += 1
		t = 0.0
		h = 0.1

		print "Starting the loop"

		while( running ):
			if not self.actions[self.r]( t ):
				print "Action Finished, changing action"
				# Action selection
				if np.random.random() < 0.5:
					print("Time to explore!")	
					self.r = np.random.randint( 0, len(self.actions) )
				else:
					self.r = np.argmax( self.Q )
						
				self.N[self.r] += 1
				t = 0.0
			
			t = t + h			
			time.sleep(0.05)

if __name__ == "__main__":
	mc = MiroController()
	mc.run()
