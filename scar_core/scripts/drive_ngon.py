#!/usr/bin/env python
"""
Moves the Neato counterclock wise around a regular polygon
based on the odometry of the robot to precalculate corners
"""

import rospy
import math
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion, rotation_matrix, quaternion_from_matrix
from geometry_msgs.msg import Pose, Twist, Vector3
from nav_msgs.msg import Odometry

import matplotlib.pyplot as plt


class ShapeDriver:

    def __init__(self, sideNum, sideLength):

        self.debugOn = True

        #States
        self.cornInd = 1 #Index of next corner
        self.corners = [] #(x,y, yaw) corners of the square to drive

        #Shape properities
        self.sides = sideNum #How many sides
        self.sideLength = sideLength #Side length of the regular polygon (meters)
        self.extAng = math.radians(180-(((self.sides-2)*180)/self.sides))
        self.linEp = .1 #How close to the target side length before turning (meters)
        self.angEp = .1 #How close to the target heading before moving (radians)

        #Robot properities
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.linVector = Vector3(x=0.0, y=0.0, z=0.0)
        self.angVector = Vector3(x=0.0, y=0.0, z=0.0)

        #ROS
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=2)
        self.rate = rospy.Rate(2)

        rospy.Subscriber("/odom", Odometry, self.setLocation)
        

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


    def calcCorners(self, startPoint):
        """
        Precalculate the corners (turning points) for the
        regular polygon to trace
        """
        if self.debugOn: print("calculating corners") 

        #First is the script's start point
        self.corners.append(startPoint)

        #Generate remaining corners
        for i in range(1,self.sides):

            #Heading to turn to upon reaching corner
            if self.corners[i-1][2]+self.extAng > math.pi:
                newTheta = self.corners[i-1][2]+self.extAng - math.pi*2
            else:
                newTheta = self.corners[i-1][2]+self.extAng

            #Set remaining corners based on heading (what quadrant the robot is facing affects +/- of x and y)
            #Up

            if self.corners[i-1][2] == 0:
                newX = self.corners[i-1][0]+self.sideLength 
                newY = self.corners[i-1][1]
            #First quadrant
            elif self.corners[i-1][2] > 0 and self.corners[i-1][2] < math.pi/2:
                newX = self.corners[i-1][0]+math.fabs(self.sideLength*math.cos(self.corners[i-1][2]))
                newY = self.corners[i-1][1]+math.fabs(self.sideLength*math.sin(self.corners[i-1][2]))
            #Left
            elif self.corners[i-1][2] == math.pi/2:
                newX = self.corners[i-1][0]
                newY = self.corners[i-1][1]+self.sideLength
            #Second quadrant
            elif self.corners[i-1][2] > math.pi/2 and self.corners[i-1][2] < math.pi:
                newX = self.corners[i-1][0]-math.fabs(self.sideLength*math.cos(self.corners[i-1][2]))
                newY = self.corners[i-1][1]+math.fabs(self.sideLength*math.sin(self.corners[i-1][2]))
            #Down
            elif self.corners[i-1][2] == math.pi:
                newX = self.corners[i-1][0]-self.sideLength
                newY = self.corners[i-1][1]
            #Third quadrant
            elif self.corners[i-1][2] > -math.pi and self.corners[i-1][2] < -math.pi/2:
                newX = self.corners[i-1][0]-math.fabs(self.sideLength*math.cos(self.corners[i-1][2]))
                newY = self.corners[i-1][1]-math.fabs(self.sideLength*math.sin(self.corners[i-1][2]))
            #Right
            elif self.corners[i-1][2] == -math.pi/2:
                nexX = self.corners[i-1][0]
                newY = self.corners[i-1][1]-self.sideLength
            #Fourth quadrant
            elif self.corners[i-1][2] > -math.pi/2 and self.corners[i-1][2] < 0:
                newX = self.corners[i-1][0]+math.fabs(self.sideLength*math.cos(self.corners[i-1][2]))
                newY = self.corners[i-1][1]-math.fabs(self.sideLength*math.sin(self.corners[i-1][2]))

            self.corners.append((newX, newY, newTheta))


    def stop(self):
        """
        Stops motion on the robot
        """
        if self.debugOn: print("stop")

        self.publishVelocity(0.0, 0.0)


    def driveForward(self):
        """
        Drives the robot forward until the next corner is reached.
        """
        if self.debugOn: print("forward")

        self.publishVelocity(0.25, 0.0)

        while not rospy.is_shutdown():
            #Has the robot reached the next corner?
            if(self.x > self.corners[self.cornInd][0]-self.linEp and self.x < self.corners[self.cornInd][0]+self.linEp):
                if(self.y > self.corners[self.cornInd][1]-self.linEp and self.y < self.corners[self.cornInd][1]+self.linEp):
                    self.stop()
                    return


    def turn(self):
        """
        Turns the robot in place until it reaches the correct heading to reach the next corner
        """
        if self.debugOn: print("turn")

        self.publishVelocity(0.0, 0.2)

        while not rospy.is_shutdown():
            #Has the robot reached the correct heading to go to the next corner?
            if (self.theta > self.corners[self.cornInd][2]-self.angEp) and (self.theta < self.corners[self.cornInd][2]+self.angEp):
                self.stop()
                return


    def visualize(self):
        """
        Plots calculated corners to ensure regular polygon is produced
        Used for debugging algorithim
        """
        x = []
        y = []
        for c in self.corners:
            print("%f\t%f\t%f" %(c[0],c[1],c[2]))
            x.append(c[0])
            y.append(c[1])

        plt.ylim(-5,5)
        plt.xlim(-5,5)
        plt.scatter(x, y)
        plt.show()


    def run(self):
        """
        Compare current position to corner locations to determine when to turn
        """
        if self.debugOn: print("drive")

        #Allow time to get first location 
        rospy.sleep(1)

        self.calcCorners((self.x, self.y, self.theta))

        for i in range(0, self.sides):
            self.driveForward()
            self.turn()
            
            #Compare to next corner
            if self.cornInd == self.sides -1:
                self.cornInd = 0
            else:
                self.cornInd += 1


        
        if self.debugOn: self.visualize()
        return 0
        


if __name__ == '__main__':
    shapeDriver = ShapeDriver(4,1)
    try:
        shapeDriver.run()
    except rospy.ROSInterruptException:
        shapeDriver.stop()