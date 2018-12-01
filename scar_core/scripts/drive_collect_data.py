#!/usr/bin/env python2

"""
Collect data as the robot drives around the object
"""

import cv2
import os
from cv_bridge import CvBridge

from icp import ICP

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

def applyTransformToPoints(T, points):
    # points = Nx2
    # T = [RT
    #      01]
    return np.dot(points, T[:2,:2].T) + T[:2,2]

class dataCollector:

    def __init__(self):
        # define constant parameters
        self.map_w = 5.0 #5x5 physical map
        self.map_h = 5.0
        self.map_res = 0.02

        self.seen_thresh = 3.0
        self.sensor_radius = 5.0

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

        #Map 1
        #Define a grid resolution to easily check if a point is in map
        self.map_res = .02
        self.seen_thresh = 3 #How many times a point must be seen to be included
        n = int(np.ceil(self.map_h / self.map_res))
        m = int(np.ceil(self.map_w / self.map_res))
        self.map = np.zeros(shape=(n,m),dtype=np.float32)

        #Map 2
        #Define a dictionary whose keys are coordinate tuples and values
        #are number of times seen. Faster because only checks points that
        #were actually seen but never gets rid of points
        self.map_dict = {}
        self.true_map = {}

        self.map_to_odom = np.zeros(shape=3)

        self.bridge = CvBridge()

        #ICP
        self.icp = ICP()

        #Ben
        self.data_path = "/home/bziemann/data/"
        #Jaime
        #self.data_path = "/tmp/"
        
        #ROS2/.02
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=2)
        self.vizPub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
        rospy.init_node('data_collection')
        rospy.Subscriber("/stable_scan", LaserScan, self.checkLaser)
        rospy.Subscriber("/projected_stable_scan", PointCloud, self.checkPoints)
        rospy.Subscriber("/odom", Odometry, self.setLocation)
        rospy.Subscriber('/camera/image_raw', Image, self.setImage)

    def realToMap(self, point):
        """
        Takes a point from the projected scan and converts it
        to our internal map

        point: rosmsg from pointCloud containing x,y data odometry for LIDAR
        scans
        """

        #Get rid of out of bound points
        if point[0] > 2.5 or point[0] < -2.5:
            return (None,None)
        if point[1] > 2.5 or point[1]<-2.5:
            return (None,None)
        
        mapX = round(abs(-2.5+point.x)/self.map_res, 0)
        mapY = round(abs(2.5+point.y)/self.map_res, 0) # indices
        self.map[mapY][mapX] += 1

    def mapToReal(self, point):
        # (i,j) --> (x,y)
        # TODO : Implement
        pass

    def realToMap2(self, point):
        """
        Take a point from the projected scan and add it to
        our dictionary map

        point: rosmsg from pointCloud containing x,y data odometry for LIDAR
        scans
        """
        px = round(point.x, 2) #TODO : make decimal points configurable
        py = round(point.y, 2)
        #Already seen
        if (px,py) in self.map_dict:
            self.map_dict[(px,py)] += 1
            #Already visualized point
            if (px,py) in self.true_map:
                return
            #Create visualized point
            elif self.map_dict[(px,py)] > self.seen_thresh:
                self.true_map[(px, py)] = None
        #New point
        else:
            self.map_dict[(px,py)] = 1

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
            #self.realToMap(p)
            #self.realToMap2(p)
        Deprecated in favor of projected stable scan

        Pulls laser scan data

        Scan: rosmsg with LIDAR Scan data

        TOOD set interoior angle where we know the object will be to reduce scans
        """
        self.old_ranges = self.ranges
        self.ranges = scan.ranges

    def queryPoints(self):
        """
        Points from the map that we care about matching with the incoming scan.
        """
        center = (self.x, self.y)
        
        # mapPoints v1
        # map_points_v1 = np.where((self.map>=self.seen_thresh)) # --> (i,j) indices
        # map_points_v1 = [self.mapToReal(p) for p in map_points_v1]

        # mapPoints v2
        # TODO : make the distance configurable although it's going to be mostly the same
        map_points_v2 = [k for k in self.map_dict.iteritems() if
                (np.linalg.norm(np.subtract(k, center)) < 5.0) and 
                (v > self.seen_thresh)
                ]

        return np.asarray(map_points_v2, dtype=np.float32)

    def checkPoints(self, msg):
        """
        Time matched LIDAR. Produces much better scans, however not all LIDAR
        is collected

        TODO:
        set interior angle where we know the object will be to reduce scans
        correct points with ICP before adding them to map
        """
        self.old_points = self.points
        self.points = []
        # TODO : update the points based on the computed offset between map->odom
        for p in msg.points:
            self.points.append(np.float32(p.x,p.y))

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
        """
        Pull image data from camera

        img: rosmsg containing image grabbed from camera
        """

        img = self.bridge.imgmsg_to_cv2(img, desired_encoding="bgr8")
        #img = PImage.open(img)
        img = cv2.resize(img, (160, 120), interpolation=cv2.INTER_AREA)
        self.img = img
        
    def update_current_map_to_odom_offset_estimate(self, t):
        pass
        
    def run(self):
        #Begin visualization
        pd = PointDumper(viz=True)
        
        #Generate data for debugging
        filename = os.path.join(self.data_path, 'data.csv')
        f = open(filename, "w+")
        f.write("x,y,theta,scan\n")
        count = 0

        #To prevent image overflow 
        image_enabled = False

        while not rospy.is_shutdown():
            
            #Make the neato move in circle of desired radius
            #r = .5
            #linX = 0.1
            #angZ = linX/r
            #self.publishVelocity(linX, angZs)

            #Don't do anything if there is no data to work with
            if (not len(self.points)==0) and (not len(self.old_points)==0):

                #Break if no image data
                if image_enabled and np.size(self.img) == 0:
                    continue

                # query the points in the map and perform ICP on them
                map_points = self.queryPoints()

                #ICP transform
                trans, diff, num_iter = self.icp.icp(self.points, map_points)

                # update the offset
                self.update_current_map_to_odom_offset_estimate(trans)

                # update the map
                points_t = applyTransformToPoints(trans, self.points)
                [self.realToMap2(p) for p in points_t]

                #Online visualization
                rospy.loginfo_throttle(1.0, 'System Online : ({},{},{})'.format(self.x,self.y,self.theta))

                # pd.proc_frame(self.x, self.y, self.theta, self.ranges)
                pd.visualize(map_points[:,0], map_points[:,1])

                line = str(self.x)+","+str(self.y)+","+str(self.theta)+","+str(self.ranges)[1:-1]+"\n"
                f.write(line)

                if image_enabled:
                    fileNum = self.data_path+"img" +str(count) +".png"
                    cv2.imwrite(fileNum,self.img)
                    count += 1 
                self.ranges=[]


if __name__ == "__main__":
    dc = dataCollector()
    dc.run()
