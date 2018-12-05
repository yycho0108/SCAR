# SCAR

Final Project for Olin Computational Robotics FA2018: SCanning And Reconstruction

- [Ben Ziemann](https://github.com/zneb97)
- [Yoonyoung Cho](https://github.com/yycho0108)

## Blog Post #2 - 12/4/18

Continuing on from our previous post, we went about applying the result of the Iterative Closest Point (ICP) algorithm we are using to both the points and the robot for a better map and better position estimate in the map.<br>

ICP for our points 2-coordinate positions (x,y) naturally results in a 3x3 homogeneous transformation matrix. Originally we were basing the transformation on a mapping from just the previous scan to the most current scan. During this week we chose to instead build up a set of points to define the entirety of the map, giving us a more data to compare against and allowing us to have a more accurate output. With this additional data also came additional computational cost though so we designed out map structure to minimize this. We explored two options.<br>

First was a 2D array, acting as a grid analog for the map. Each pair of indices was a point on the map (at a given resolution). The value at those indices indicated how many times it had been seen as we wanted to not include any one-off scan noise or temporary drift. We ended up moving to a different approach as our second pass as this one was limiting in that the grid could only start at a specific size and making it larger dynamically would be difficult. Corresponding x and y coordinates between map, robot, and indices meant it was not easily followable.<br>

The second pass used similar concepts but instead made use of dictionaries with coordinate tuples as keys and the times seen as the value. This allowed us much easier accessing of the data while not needing to allocate memory for points that never would have been seen anyways. The coordinates also didn’t need to undergo scaling and still make sense in context with the odometry.<br>

We have begun finalizing our logic-level implementation of how the ICP-correction will apply to our long-term pose estimates and scan quality, where the workflow is roughly as follows:<br>

* Query the Map for scan-matching candidates
* Apply ICP to compute plausible transform offset
* Update the Map
* Update the <b>[map→odom]</b> Transformation based on the computed offset

The breakdown of the pipeline into such discrete steps allow for a layer of abstraction to experiment with different implementations at each stage. Specifically, the current areas of exploration includes:<br>
* Intelligent Map queries with Ray-casting
    * Avoiding ICP being applied to a map with key features being thicker than they should be.
* Sparse vs. dense data-structures for map representation
* Weighted transform updates with ICP confidence estimates
    * Clearing points based on how often we haven’t seen them when they still should be in our LIDAR range
* Loop Closure
    * Defining landmarks upon initialization
 <br>

Current issues we are facing are the initialization not running as smoothly as we would have hoped, with it often jumping radically, causing more sets of points for ICP to align with, causing the wrong transformation between points. Additionally even during a good initialization, there is still some jitter that if left to sit will slowly expand outwards as ICP occasionally matches to that. This builds out larger and larger padding as shown below:<br>

![alt text](https://github.com/yycho0108/SCAR/blob/master/images/init.png "Initialization")
<br>
Figure 1: Initialization. Notice the slight jitter and extra points
<br>

![alt text](https://github.com/yycho0108/SCAR/blob/master/images/padding.png "Jitter causes ICP to match in increasing padding")
<br>
Figure 2: Jitter causes ICP to match in increasing padding.
<br>

![alt text](https://github.com/yycho0108/SCAR/blob/master/images/padding2.png "After staying stationary for several minutes")
<br>
Figure 3: After staying stationary for several minutes
<br>

## Blog Post #1 - 11/30/18

For the first two weeks of the this project we worked on implementing a basic driving and visualization script as our intended first pass. We wanted to understand the data that we were working with before diving too deeply into a particular implementation, which is why we ended up constructing a number of visualizations that showed statistics such as an accumulated overlay of data, or the histogram of the number of incoming scans at each point in time.<br>

In order to get an early start on the simulation development, which was one of our learning goals, we developed upon the existing simulator with the appropriate modifications for positioning our sensors.<br>


![alt text](https://github.com/yycho0108/SCAR/blob/master/images/neato.png "Simulation")
<br>
Fig 1. BotVac based Neato model in Gazebo Simulator
<br>
While the simulation had not been used extensively at this point, the presence of ground truth information as we’re testing, along with the highly deterministic and controlled testing environment, will provide an ideal benchmark for our development in the project.

![alt text](https://github.com/yycho0108/SCAR/blob/master/images/maps.png "Projected maps")
<br>
Figure 2: Projected map with robot moving quickly, ~1m/s, (left) and moving slowly, ~.3m/s (right).
<br>

Fig 2 depicts maps created from driving around a trash can near the front of room AC 109. The left is at a higher speed with significant movement we believed is caused by drift. The right is moving at a much slower speed to combat the problem of drift but still has an overlaying problem which we believe is due to the combination of odometry and LIDAR forming a transformation that is off. We plan on addressing these issues using an ICP algorithm to better map data to existing data, rather than depend solely on odometry and the scan. <br>

One particular concern on our part was the repeatability of the scan data over the course of the navigation trajectory, such that a previously detected point which represents the object will be detected again at the same global coordinate as the prior location. In order to characterize the overall tolerance on the amount of perturbation that the transform would undergo across spatio-temporal gaps, we observed the stability of the laser scan data over a short driving time and concluded that we will need to account for this alogirthimically, hence our project.

To mitigate this we took a look different factors and began implementing solutions.
First was the reliability of the Lidar scan data when using projected scan. Creating a histogram, we found that while driving about the open front of room AC 109 we had relatively few points, only about 60 or about 1/6th the possible number of points the LIDAR could be picking up. This information gave as a better intuition about how to go about implementing and effectively using ICP going forward.
 
![alt text](https://github.com/yycho0108/SCAR/blob/master/images/histo.png "Histogram of LIDAR Scan")
<br>
Figure 3: Hisogram of successful (non NaN) LIDAR scans.
<br>

We then began an exploration of an iterative closest points (ICP) algorithm so as to better relate one set of scan data to the previous scan. We spent some time understanding the output of a typical ICP algorithm (homogeneous transformation matrices) as well as formatting our data to work with it. Knowing we are not going to get the same number of points with every scan, we decided on only using sampling of each scan. Additionally using generated data we did some exploration how much a data set can be off by before the output transformation matrix is significantly affected.

