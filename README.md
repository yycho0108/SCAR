# SCAR

Final Project for Olin Computational Robotics FA2018: SCanning And Reconstruction

- [Ben Ziemann](https://github.com/zneb97)
- [Yoonyoung Cho](https://github.com/yycho0108)

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

