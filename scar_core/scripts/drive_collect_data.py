#!/usr/bin/env python2

"""
Collect data as the robot drives around the object
"""

import cv2
import os
from cv_bridge import CvBridge

from drive_ngon import ShapeDriver
from icp import icp

from PIL import Image as PImage
import matplotlib.pyplot as plt
import numpy as np
import pandas
import csv

import rospy
from sensor_msgs.msg import Image, LaserScan, PointCloud
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion, rotation_matrix, quaternion_from_matrix
from geometry_msgs.msg import Pose, Twist, Vector3
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker

from scar_core.point_dumper import PointDumper

class dataCollector:

    def __init__(self):

        self.debugOn = False

        #Robot properities
        self.linVector = Vector3(x=0.0, y=0.0, z=0.0)
        self.angVector = Vector3(x=0.0, y=0.0, z=0.0)

        self.x = 0
        self.y = 0
        self.theta = 0
        self.ranges = [] #From stable scan
        self.old_ranges = []
        self.points = [] #From projected stable scan
        self.old_points = []
        self.img = []

        self.bridge = CvBridge()

        #ICP
        self.icp = icp()

        #Ben
        self.data_path = "/home/bziemann/data/"
        #Jaime
        #self.data_path = "/tmp/"
        
        #ROS
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=2)
        self.vizPub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
        rospy.init_node('data_collection')
        rospy.Subscriber("/stable_scan", LaserScan, self.checkLaser)
        rospy.Subscriber("/projected_stable_scan", PointCloud, self.checkPoints)#TODO
        rospy.Subscriber("/odom", Odometry, self.setLocation)
        rospy.Subscriber('/camera/image_raw', Image, self.setImage)


    def publishVelocity(self, linX, angZ):
        """
        Publishes velocities to make the robot move

        linX is a floating point between 0 and 1 to control the robot's x linear velocity
        angZ is a floating point between 0 and 1 to control the robot's z angular velocity
        """
        if self.debugOn: print("publishing")

        self.linVector.x = linX
        self.angVector.z = angZ
        self.pub.publish(Twist(linear=self.linVector, angular=self.angVector))


    def checkLaser(self, scan):
        """
        Pulls laser scan data

        TOOD set interoior angle where we know the object will be to reduce scans
        """
        self.old_ranges = self.ranges
        self.ranges = scan.ranges


    def checkPoints(self, msg):
        """
        Time matched LIDAR. Produces much better scans, however not all LIDAR
        is collected
        TOOD set interoior angle where we know the object will be to reduce scans
        
        """
        self.old_points = self.points
        self.points = np.transpose([(p.x, p.y) for p in msg.points])
      

    def setLocation(self, odom):
        """
        Convert pose (geometry_msgs.Pose) to a (x, y, theta) tuple
        Constantly being called as it is the callback function for this node's subscription

        odom is Neato ROS' nav_msgs/Odom msg composed of pose and orientation submessages
        """

        pose = odom.pose.pose
        orientation_tuple = (pose.orientation.x,
                                pose.orientation.y,
                                pose.orientation.z,
                                pose.orientation.w)
        angles = euler_from_quaternion(orientation_tuple)
        self.x = pose.position.x
        self.y = pose.position.y
        self.theta = angles[2]
        return (pose.position.x, pose.position.y, angles[2])


    def setImage(self, img):

        img = self.bridge.imgmsg_to_cv2(img, desired_encoding="bgr8")
        #img = PImage.open(img)
        img = cv2.resize(img, (160, 120), interpolation=cv2.INTER_AREA)
        self.img = img

        
    def run(self):
        pd = PointDumper(viz=True)
        
        filename = os.path.join(self.data_path, 'data.csv')
        f = open(filename, "w+")
        f.write("x,y,theta,scan\n")
        count = 0

        image_enabled = False
        while not rospy.is_shutdown():
            
            #self.publishVelocity(0.1,0.15)

            if (not len(self.points)==0) and (not len(self.old_points)==0):
                if image_enabled and np.size(self.img) == 0:
                    continue

                #icp calculations
                # print(self.old_points)
                # print("The length is %i" %len(self.old_points))
                # print(self.old_points[200:340]) #TODO why isn't this giving errors?


                #Fix the difference in # of points
                past_points = np.array(self.old_points)
                curr_points = np.array(self.points)
                icp_range = min(past_points.shape[1],curr_points.shape[1])

                edge_fuzz = 2 #How much of the edge to ignore
                if (past_points.shape[1] > curr_points.shape[1]):
                       
                        past = past_points[:, edge_fuzz:curr_points.shape[1]-edge_fuzz]
                        curr = curr_points[:, edge_fuzz:-edge_fuzz]

                else:
                        past = past_points[:, edge_fuzz:-edge_fuzz]
                        curr = curr_points[:, edge_fuzz:past_points.shape[1]-edge_fuzz]
                        
                trans, diff, num_iter = self.icp.icp(past, curr)

                # online visualization
                rospy.loginfo_throttle(1.0, 'System Online : ({},{},{})'.format(self.x,self.y,self.theta))

                # pd.proc_frame(self.x, self.y, self.theta, self.ranges)
                pd.visualize(self.points[0], self.points[1])
                line = str(self.x)+","+str(self.y)+","+str(self.theta)+","+str(self.ranges)[1:-1]+"\n"
                f.write(line)

                if image_enabled:
                    fileNum = self.data_path+"img" +str(count) +".png"
                    cv2.imwrite(fileNum,self.img)
                    count += 1 
                self.ranges=[]
            #v=wr
            #Radius of 1 meter

    def icpTest(self):
        x=1
            


if __name__ == "__main__":
    dc = dataCollector()
    dc.run()
