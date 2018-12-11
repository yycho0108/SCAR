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
import math
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
from scan_map import DenseMap, SparseMap

def applyTransformToPoints(T, points):
    """
    Apply a homogenous transformation to a 2xn matrix
    """
    return np.dot(points, T[:2,:2].T) + T[:2,2]

def R2(x):
    c = np.cos(x)
    s = np.sin(x)
    return np.reshape([c,-s,s,c], (2,2))


def truncate(f, n):
    '''
    Truncates/pads a float f to n decimal places without rounding
    Deals with storing keys and values in dictionaries

    From user David Z on Stackover flow
    https://stackoverflow.com/questions/783897/truncating-floats-in-python
    '''
    s = '%.12f' % f
    i, p, d = s.partition('.')
    return float('.'.join([i, (d+'0'*n)[:n]]))

class dataCollector:

    def __init__(self, map_args=None,):

        #Debugging flags
        self.prints_enabled = False
        self.images_enabled = False

        #Robot properities
        self.linVector = Vector3(x=0.0, y=0.0, z=0.0)
        self.angVector = Vector3(x=0.0, y=0.0, z=0.0)
        self.lidar_range = 5.0
            #From odom
        self.x_odom = 0
        self.y_odom = 0
        self.theta_odom = 0
            #From stable scan
        self.ranges = []
        self.old_ranges = []
            #From projected stable scan
        self.points = []
        self.old_points = []
            #From /raw_camera_image
        self.img = None
        self.bridge = CvBridge()

        

        #Built up map
        #Define a dictionary whose keys are coordinate tuples and values
        #are number of times seen
        self.map_dict = {}
        self.map_final ={}
        self.seen_thresh = 3
        self.map_resolution = 0.01 #Meters
        self.angle_res = .001 #Radians
        self.trunc_val = 4

        #Flag variables
        self.stop_getting_points_flag = 0

        #Counters
            #How many times points have been sampled
        self.point_cycle_counter = 0

        #ICP
        self.icp = ICP()
            #Point and robot location offset based on ICP, (x,y,theta)
        self.offset = np.zeros(shape=3)

        #Visualizer
        self.point_dumper1 = PointDumper(viz=True)
        self.point_dumper2 = PointDumper(viz=True)
        self.point_dumper3 = PointDumper(viz=True)

        #Where to save data
        #Ben
        #self.data_path = "/home/bziemann/data/"
        #Jaime
        self.data_path = "/tmp/"
        
        #ROS
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=2)
        rospy.init_node('data_collection')
        rospy.Subscriber("/stable_scan", LaserScan, self.checkLaser)
        rospy.Subscriber("/projected_stable_scan", PointCloud, self.checkPoints)
        rospy.Subscriber("/odom", Odometry, self.setLocation)
        rospy.Subscriber('/camera/image_raw', Image, self.setImage)

#Publishing functions
    def publishVelocity(self, linX, angZ):
        """
        Publishes velocities to make the robot move

        linX is a floating point betwmapToRealeen 0 and 1 to control the robot's x linear velocity
        angZ is a floating point between 0 and 1 to control the robot's z angular velocity
        """
        if self.debugOn: print("publishing")

        self.linVector.x = linX
        self.angVector.z = angZ
        self.pub.publish(Twist(linear=self.linVector, angular=self.angVector))


#Callback functions
    def checkLaser(self, scan):
        """
        Pulls laser scan data
        """
        if self.stop_getting_points_flag:
            return
        else:
            self.old_ranges = self.ranges
            self.ranges = scan.ranges


    def checkPoints(self, msg):
        """
        Pulls laser scan data in the form of (x,y) points in relation to odometry
        """
        if self.stop_getting_points_flag:
            return
        else:
            self.old_points = self.points
            self.points = [[p.x, p.y] for p in msg.points]
            self.point_cycle_counter += 1

    def setLocation(self, odom):
        """
        Pulls odometry data
        Convert pose (geometry_msgs.Pose) to a (x, y, theta) tuple
        """
        pose = odom.pose.pose
        orientation_tuple = (pose.orientation.x,
                                pose.orientation.y,
                                pose.orientation.z,
                                pose.orientation.w)
        angles = euler_from_quaternion(orientation_tuple)
        self.x_odom = pose.position.x
        self.y_odom = pose.position.y
        self.theta_odom = angles[2]
        return (pose.position.x, pose.position.y, angles[2])


    def setImage(self, img):
        """
        Pull image data from camera
        """
        if self.images_enabled:
            img = self.bridge.imgmsg_to_cv2(img, desired_encoding="bgr8")
            img = cv2.resize(img, (160, 120), interpolation=cv2.INTER_AREA)
            self.img = img


#Work functions
    def update_offset(self, t):
        """
        Update the robot's offset based on ICP result. Used in estimated position offset from odometry

        t = homogenuous transformation matrix output of ICP
        """
        #rotation_trans = np.dot(t[:2,:2], np.array([[self.x, self.y]]).T)
        #self.map_to_odom = np.array([t[0][2], t[1][2], np.arctan2(rotation_trans[1],rotation_trans[0])])

        dx0, dy0, dh0 = self.offset # what it used to be

        # current error
        ddx, ddy = t[:2,2]
        ddh = math.atan2(t[1,0], t[0,0])

        # apparently?
        dx1, dy1 = R2(ddh).dot([dx0,dy0]) + [ddx,ddy]
        dh1 = (dh0 + ddh)

        self.offset = np.asarray([dx1,dy1,dh1])

    def round_to_res(self, val, res):
        if val%res > res/2:
            return truncate(math.ceil(val/res)*res,self.trunc_val)
        else:
            return truncate(math.floor(val/res)*res,self.trunc_val)

    def dist_to_point(self, point):
        """
        Get distance from robot to point
        """
        center = (self.x_odom, self.y_odom)
        trans_center = self.apply_offset(center)
        return math.sqrt(math.pow(trans_center[0]-point[0],2) + math.pow(trans_center[1]-point[1],2))


    def filterPoints(self, point):
        """
        Take a point from the projected scan and add it to
        our dictionary maps
        """

        #Round to nearest increment of map resolution
        px = self.round_to_res(point[0], self.map_resolution)
        py = self.round_to_res(point[1], self.map_resolution)
        coordinates =(px,py)

        #Filtering for disance
        if self.dist_to_point(coordinates) <= self.lidar_range:

            #Get data
            dist = self.dist_to_point(coordinates)
            dist = self.round_to_res(dist, self.map_resolution)
            angle = self.angle_to_point(coordinates)[0]
            angle = self.round_to_res(angle, self.angle_res)

            #Already seen
            if (px,py) in self.map_dict:
                self.map_dict[coordinates] = (dist, angle, self.map_dict[coordinates][2]+1)

                #Already visualized point
                if coordinates in self.map_final:
                    self.map_final[coordinates] = (dist, angle, self.map_final[coordinates][2]+1)

                #Filtering for seen enough
                elif self.map_dict[coordinates][2] > self.seen_thresh:
                    self.map_final[coordinates] = (dist, angle, self.map_dict[coordinates][2])

            #New point
            else:
                self.map_dict[coordinates] = (dist, angle, 1)


    def map_to_matrix(self, m):
        """
        Convert map to matrix of points
        """
        map_curr = [coordinates for (coordinates,x) in m.iteritems() if
            self.dist_to_point(coordinates) <= self.lidar_range]

        return map_curr


    def apply_offset(self, points):
        """
        Apply the current offset to points
        """
        T_o2m = np.eye(3)
        T_o2m[:2,:2] = R2(self.offset[-1])
        T_o2m[:2,2]  = self.offset[:2]

        trans_points = applyTransformToPoints(T_o2m, points)
        return trans_points


    def angle_to_point(self, point):
        """
        Get angle from robot to point
        """
        center = (self.x_odom, self.y_odom)
        trans_center = self.apply_offset(center)

        c, s = np.cos(self.theta_odom), np.sin(self.theta_odom)
        trans_mat = np.array(((c,-s), (s, c)))
        offset_vec = np.array([0,1]).T
        offset_vec =np.dot(trans_mat, offset_vec)

        angle = math.atan2(point[1], point[0])-math.atan2(offset_vec[1], offset_vec[0])
        if (angle < 0):
            angle += 2 * math.pi
        angleDeg = np.degrees(angle)
        return angle, angleDeg


    def update_map(self):
        """
        Updates the final map with new angles and distances for all points in it
        """
        for (coordinates,x) in self.map_final.iteritems():
            dist = self.dist_to_point(coordinates)
            dist = self.round_to_res(dist, self.map_resolution)
            angle = self.angle_to_point(coordinates)[0]
            angle = self.round_to_res(angle, self.angle_res)    
            self.map_final[coordinates] = (dist, angle, self.map_final[coordinates][2])


        #Might more general map might nnot need this as this is only used in raycast which is only used to generate
        #the map for comparing ICP and then dropped
        for (coordinates,x) in self.map_dict.iteritems():
            dist = self.dist_to_point(coordinates)
            dist = self.round_to_res(dist, self.map_resolution)
            angle = self.angle_to_point(coordinates)[0]
            angle = self.round_to_res(angle, self.angle_res)    
            self.map_dict[coordinates] = (dist, angle, self.map_dict[coordinates][2])


    def raycast(self, points):
        """
        Compare current scan points and existing map points
        to determine which points should not be considered when
        decaying probabilities
        """

        #Update Map 
        # {(x,y):(dist,ro
        self.update_map()


        #Construct angle map
        # {rounded_ang:[(x1,y1),(x2,y2)...]}
        map_final_angle = {}
        for (coordinates, data) in self.map_final.iteritems():
            #Filter for distance
            if self.dist_to_point(coordinates) <= self.lidar_range:
                if data[1] in map_final_angle:
                    map_final_angle[data[1]].append(coordinates)
                else:
                    map_final_angle[data[1]] = [coordinates]


        #Construct current scan maps
        # {(x,y):(dist,rounded_angle)}
        map_scan = {}
        # {rounded_ang:(x1,y1)}
        map_scan_angle = {}

        for point in points:
            #Round points to map res
            px = self.round_to_res(point[0], self.map_resolution)
            py = self.round_to_res(point[1], self.map_resolution)
            point = (px,py)

            dist = self.dist_to_point(point)
            dist = self.round_to_res(dist, self.map_resolution)
            angle = self.angle_to_point(point)[0]
            angle = self.round_to_res(angle, self.angle_res)

            if point in map_scan:
                map_scan[point] = (min(map_scan[point][0], dist), angle)
            else:
                map_scan[point] = (dist, angle)


            if angle in map_scan_angle:
                map_scan_angle[angle].append(point)
            else:
                map_scan_angle[angle] = [point]
        

        # #Create list of points to ignore that are "behind" other points
        ignore_points = [] 
        for (coordinates, data) in map_scan.iteritems():
            if data[1] in map_final_angle:
                for p in map_final_angle[data[1]]:
                    #Compare distances to avoid directly overlapping problems
                    if self.map_final[p][0] >= data[0]:
                        ignore_points.append(p)


        #Create a list of points on the map to consider for ICP
        map_to_consider = {}
        for (coordinates, data) in self.map_final.iteritems():
            #Filter for sight
            if coordinates in ignore_points:
                continue
            #Filter for distance
            elif self.map_final[coordinates][0] >= self.lidar_range:
                continue
            
            map_to_consider[coordinates] = None
        
        return map_to_consider



    def run(self):

        #Setup data file
        filename = os.path.join(self.data_path, 'data.csv')
        f = open(filename, "w+")
        f.write("x,y,theta,scan\n")
        img_count = 0

        print("########### SETUP - PLEASE DON'T MOVE ROBOT ###########")

        #Build an initial map
        while (self.point_cycle_counter < 10) and (not rospy.is_shutdown()):
            print("Cycle ", self.point_cycle_counter)
            if (not len(self.points)==0):
                for point in self.points:
                    self.filterPoints(point)

        print("########### SETUP COMPLETE ###########")

        #Begin using ICP
        while (not rospy.is_shutdown()):

            # self.stop_getting_points_flag=True
            # rospy.sleep(.01)
            points = self.points
            map_curr = self.map_final
            # self.stop_getting_points_flag=False

            if (not len(points)==0):
                #Avoid no image errors
                if self.images_enabled and np.size(self.img) == 0:
                    continue

                # Apply offset to points
                offset_points = self.apply_offset(points)

                #Determine which points should be considered for ICP
                map_curr = self.raycast(offset_points)
                map_curr = self.map_to_matrix(map_curr)

                #Get ICP trasnformation from (offset?) points to final map
                trans, diff, num_iter = self.icp.icp(np.asarray(offset_points), np.asarray(map_curr))
                self.update_offset(trans)

                #Apply ICP transformation to current points
                trans_points = np.asarray(applyTransformToPoints(trans, offset_points))

                #Add adjusted points to the map
                for p in trans_points:
                    self.filterPoints(p)

                #Visualize current maps
                #Remake the map matrix to be used for visualizing
                map_curr = self.map_to_matrix(self.map_final)
                map_curr = np.asarray(map_curr)
                points = np.asarray(points)
                center = (self.x_odom, self.y_odom)
                trans_center = self.apply_offset(center)

                if (len(map_curr) > 0) and (len(self.points) > 0) and (len(trans_points) > 0):
                    self.point_dumper1.visualize(map_curr[:,0], map_curr[:,1], clear=True, label='map')
                    self.point_dumper2.visualize(points[:,0], points[:,1], clear=True, label='scan')
                    self.point_dumper3.visualize(trans_points[:,0], trans_points[:,1], clear=False, draw=True, label='transformed')
                    #self.point_dumper1.visualize([self.x_odom], [self.y_odom], clear=False, draw=True, label='robot')
                    #self.point_dumper2.visualize([self.x_odom], [self.y_odom], clear=False, draw=True, label='robot')
                    self.point_dumper3.visualize([trans_center[0]], [trans_center[1]], clear=False, draw=True, label='robot')
                
                #Save images
                if self.images_enabled:
                    fileNum = self.data_path+"img" +str(count) +".png"
                    cv2.imwrite(fileNum,self.img)
                    self.img_count += 1 


if __name__ == "__main__":
    dc = dataCollector()
    dc.run()