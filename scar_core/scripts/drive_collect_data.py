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
from collections import defaultdict

from scar_core.point_dumper import PointDumper

from scan_map import DenseMap, SparseMap, ProbMap
import tf


def applyTransformToPoints(T, points):
    print(points)
    """
    Apply a homogenous transformation on a list of points

    T - 2D homogenous transformation matrix, output of ICP
    points - list of points [[x,y]]

    """
    return np.dot(points, T[:2,:2].T) + T[:2,2]


def R2(x):
    """
    Create a rotation matrix

    x - angle of rotation (radians)
    """
    c = np.cos(x)
    s = np.sin(x)
    return np.reshape([c,-s,s,c], (2,2))


def convert_to_polar(pts, origin):
    """
    Convert current map to polar coordinates, with the robot at the origin
    """

    op = np.reshape(origin[:2], (1,2))
    oh = origin[-1]

    pts = (pts - op).dot(R2(oh))

    r = np.linalg.norm(pts, axis=-1)
    h = np.arctan2(pts[:,1], pts[:,0])

    return np.stack([r,h], axis=-1)


class ScanGUI(object):

    def __init__(self):
        self.fig_ = plt.figure()
        self.ax0_ = self.fig_.add_subplot(1,2,1)
        self.ax1_ = self.fig_.add_subplot(1,2,2)


    def __call__(self,
            scan,
            map,
            path,
            origin):

        self.ax0_.cla()
        self.ax1_.cla()

        self.ax0_.plot(map[:,0], map[:,1], '.', label='map')
        self.ax0_.plot(scan[:,0], scan[:,1], '+', label='scan')
        self.ax0_.plot(path[:,0], path[:,1], '--', label='path')

        # self
        x,y,h = origin
        c, s = np.cos(h), np.sin(h)
        self.ax0_.quiver([x],[y],[c],[s],
                scale_units='xy',
                angles='xy'
                )

        # sensor radius
        sen_h = np.linspace(-np.pi, np.pi)
        sen_c, sen_s = np.cos(sen_h), np.sin(sen_h)
        sen_x, sen_y = x + 5.0 * sen_c, y + 5.0 * sen_s
        self.ax0_.plot(sen_x, sen_y, ':', label='range')


    def show(self):
        self.ax0_.legend()
        self.ax0_.set_aspect('equal', 'datalim')
        self.fig_.canvas.draw()
        plt.pause(0.001)


class dataCollector:

    def __init__(self):
        
        self.gui_ = ScanGUI()

        # map params
        self.map_w = 20.0 #5x5 physical map
        self.map_h = 20.0
        # self.map_ = SparseMap(res = 0.02)
        # self.map_ = DenseMap(w=self.map_w, h=self.map_h, res = 0.1)
        self.map_ = ProbMap(w=self.map_w, h=self.map_h, res=0.1)

        # raycast params
        self.dist_res = .01
        self.ang_res  = np.deg2rad(1.0)

        # query params
        self.sensor_radius = 5.0
        #self.seen_thresh = 2
        self.seen_thresh = 0.68 # probability! TODO: tune

        #Robot properities
        self.linVector = Vector3(x=0.0, y=0.0, z=0.0)
        self.angVector = Vector3(x=0.0, y=0.0, z=0.0)
        self.lidar_range = 5.0

        # 2d pose
        self.x = 0
        self.y = 0
        self.theta = 0
        self.path = np.empty(shape=(0,3), dtype=np.float32)

        # sensor data cache
        self.ranges = [] #From stable scan
        self.old_ranges = []
        self.points = [] #From projected stable scan
        self.old_points = []
        self.img = []

        #Offset based on ICP, (x,y,theta)
        self.map_to_odom = np.zeros(shape=3)

        self.bridge = CvBridge()

        #ICP
        self.icp = ICP()

        #Ben
        #self.data_path = "/home/bziemann/data/"
        #Jaime
        self.data_path = "/tmp/"
        
        #ROS2/.02
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=2)
        self.vizPub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
        self.tfl_ = tf.TransformListener()
        self.tfb_ = tf.TransformBroadcaster()
        
        rospy.Subscriber("/stable_scan", LaserScan, self.checkLaser)
        rospy.Subscriber("/projected_stable_scan", PointCloud, self.checkPoints)
        #rospy.Subscriber("/odom", Odometry, self.setLocation)
        rospy.Subscriber('/camera/image_raw', Image, self.setImage)


    def convert_points(self, points):
        """
        Apply offset from previous rounds to current scan

        points - the current set of points from the LIDAR scan
        """

        T_o2m = np.eye(3)
        T_o2m[:2,:2] = R2(self.map_to_odom[-1])
        T_o2m[:2,2]  = self.map_to_odom[:2]
        points = applyTransformToPoints(T_o2m, points)
        return points


    def publishVelocity(self, linX, angZ):
        """
        Publishes velocities to make the robot move

        linX - 0 and 1 to control the robot's x linear velocity (m/s)
        angZ - 0 and 1 to control the robot's z angular velocity (rad/s)
        """
        self.linVector.x = linX
        self.angVector.z = angZ
        self.pub.publish(Twist(linear=self.linVector, angular=self.angVector))


    def checkLaser(self, scan):
        """
        Deprecated in favor of projected stable scan

        Pulls laser scan data

        Scan: rosmsg with LIDAR Scan data
        """
        self.old_ranges = self.ranges
        self.ranges = scan.ranges


    def queryPoints(self):
        """
        Points from the map that we care about matching with the incoming scan.
        """
        center = (self.x, self.y)
        points = self.map_.query(
                origin = (self.x, self.y, self.theta),
                radius = self.sensor_radius,
                thresh = self.seen_thresh
                )
        return points


    def checkPoints(self, msg):
        """
        Time matched LIDAR. Produces much better scans, however not all LIDAR
        is collected
        """

        self.old_points = self.points
        points = [ [p.x, p.y] for p in msg.points]

        # convert incoming points to map frame
        self.points = self.convert_points(points)
        

    def setTFLocation(self):
        """
        Update the robot's estimated position based on offset
        """
        try:
            txn, qxn = self.tfl_.lookupTransform('odom', 'base_link', rospy.Time(0))
        except Exception as e:
            rospy.loginfo_throttle(1.0, 'Failed TF Transform : {}'.format(e))
            return

        # odom frame
        x, y = txn[0], txn[1]
        h    = tf.transformations.euler_from_quaternion(qxn)[-1]
        x, y = self.convert_points([[x, y]])[0]
        h    = h + self.map_to_odom[-1]

        self.x, self.y = (x,y)
        self.theta = h


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

        # convert to map position
        # TODO : conversion happens here, or delay?
        self.x, self.y = self.convert_points([[pose.position.x, pose.position.y]])[0]
        self.theta = angles[2] + self.map_to_odom[-1]


    def setImage(self, img):
        """
        Pull image data from camera

        img: rosmsg containing image grabbed from camera
        """
        img = self.bridge.imgmsg_to_cv2(img, desired_encoding="bgr8")
        img = cv2.resize(img, (160, 120), interpolation=cv2.INTER_AREA)
        self.img = img


    def update_current_map_to_odom_offset_estimate(self, t):
        """
        Update the robot's offset based on ICP result. Used in estimated position offset from odometry

        t = homogenuous transformation matrix output of ICP
        """
        # what it used to be
        dx0, dy0, dh0 = self.map_to_odom

        # current error
        ddx, ddy = t[:2,2]
        ddh = math.atan2(t[1,0], t[0,0])

        # update with coupled transform
        dx1, dy1 = R2(ddh).dot([dx0,dy0]) + [ddx,ddy]
        dh1 = (dh0 + ddh)
        self.map_to_odom = np.asarray([dx1,dy1,dh1])


    def raycast(self, map_points):
        """
        Perform raycasting on current map to decide what points to use in ICP
        """

        # Step 1 - Get list of map points + convert to polar coordinates centered around the robot
        rh = convert_to_polar(map_points, [self.x, self.y, self.theta])
        rads, angs = rh[:,0], rh[:,1]
        angs = np.round(angs / self.ang_res).astype(np.int32)

        # Step 2 - Put the points into bins based on specified angular resolution
        rbins = defaultdict(lambda:[])
        pbins = defaultdict(lambda:[])

        for (p,r,a) in zip(map_points, rads, angs):
            rbins[a].append(r)
            pbins[a].append(p)

        # Step 3 - Go through the points and find the coordinates of the minimum radius
        points = []
        for a in angs:
            idx = np.argmin(rbins[a])
            points.append(pbins[a][idx])

        points = np.asarray(points, dtype=np.float32) # convert to np array
        return points

    def run(self):
        
        #Generate data for debugging
        filename = os.path.join(self.data_path, 'data.csv')
        f = open(filename, "w+")
        f.write("x,y,theta,scan\n")
        count = 0

        #To prevent image overflow 
        image_enabled = False

        rate = rospy.Rate(100)

        while not rospy.is_shutdown():
            points = np.asarray(self.points)

            #Don't do anything if there is no data to work with
            if (not len(points)==0) and (not len(self.old_points)==0):
                self.points = []
                count += 1

                if image_enabled and np.size(self.img) == 0:
                    continue

                #Update robot position
                self.setTFLocation()
                map_points = self.queryPoints()

                #Online visualization
                rospy.loginfo_throttle(1.0, 'System Online : ({},{},{})'.format(self.x,self.y,self.theta))

                # logging
                line = str(self.x)+","+str(self.y)+","+str(self.theta)+","+str(self.ranges)[1:-1]+"\n"
                f.write(line)

                valid_map_points = np.empty(shape=(0,2), dtype=np.float32)
                trans = None

                try:
                    valid_map_points = self.raycast(map_points)

                    #Ensure enough data to work with
                    c_map_cnt  = len(valid_map_points) >= 20
                    c_scan_cnt = len(points) >= 20

                    #ICP
                    if c_map_cnt and c_scan_cnt:
                        trans, diff, idx, num_iter, inl = self.icp.icp(
                                points,
                                valid_map_points)

                        #Ensure no massive jumps caused by incorrect transform
                        offset_euclidean = np.linalg.norm(trans[:2,2])
                        offset_h = trans

                        c_jump = ((offset_euclidean > 0.25) and (inl < 0.75)) # unsupported jump
                        c_jump = c_jump or offset_euclidean > 1.0 # "impossible" jump
                        c_sparse = (inl < 0.5) # not enough inliers

                        if c_jump or c_sparse:
                            cause = ('jump' if c_jump else 'insufficient constraint/stability')
                            msg = 'rejecting transform due to {} : d={} p={}%' .format(
                                    cause, offset_euclidean, 100*inl)
                            print(msg)
                            trans = None

                    else:
                        trans = None
                except Exception as e:
                    print 'e', e
                    print map_points
                    print map_points.shape
                    raise e
                #Update the offset - but only if more than 5 scans have been registered so far
                # (prevent premature transform computation)

                # update the map with the transformed points
                if (trans is not None) and (count >= 5):
                    self.update_current_map_to_odom_offset_estimate(trans)
                    self.map_.update(applyTransformToPoints(trans, points),
                            origin=[self.x, self.y, self.theta]
                            )
                   

                    #Send correction to ROS TF Stream
                    m2ox, m2oy, m2oh = self.map_to_odom
                    self.tfb_.sendTransform(
                            [m2ox, m2oy, 0],
                            tf.transformations.quaternion_from_euler(0, 0, m2oh),
                            rospy.Time.now(),
                            'odom',
                            'map')
                else:
                    self.map_.update(points, origin=[self.x, self.y, self.theta])

                

               #Vizualize
                if len(map_points) > 0:
                    self.gui_(points, valid_map_points, self.path,
                            origin=[self.x,self.y,self.theta])
                    self.map_.show(ax=self.gui_.ax1_)
                    self.gui_.show()

                #Save Images
                self.path = np.concatenate([self.path, [[self.x, self.y, self.theta]] ], axis=0)
                if image_enabled:
                    fileNum = self.data_path+"img" +str(count) +".png"
                    cv2.imwrite(fileNum,self.img)
                self.ranges=[]
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node('data_collection')
    dc = dataCollector()
    dc.run()
