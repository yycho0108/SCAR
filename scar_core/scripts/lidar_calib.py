import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from matplotlib import pyplot as plt

def anorm(x):
    return (x + np.pi) % (2*np.pi) - np.pi

class LidarStats(object):
    def __init__(self):
        self.fig_, self.ax_ = plt.subplots(1,3)
        self.new_data_ = False
        self.data_ = []
        self.ang_ = []

        self.scan_sub_ = rospy.Subscriber("/stable_scan",
                LaserScan, self.scan_cb)

    def scan_cb(self, msg):
        r = np.float32(msg.ranges)
        a = np.linspace(0, 2*np.pi, 361)
        a = anorm(a)

        mask1 = np.isfinite(r)
        mask2 = (r >= msg.range_min)
        mask3 = (r < msg.range_max)
        mask4 = a > -np.deg2rad(15)
        mask5 = np.deg2rad(15) > a
        mask = np.logical_and.reduce([mask1, mask2, mask3, mask4, mask5])
        
        entry = len(r[mask])
        self.data_.extend(r[mask])

        a = a[mask]
        c = np.cos(a)
        s = np.sin(a)
        x = c * r[mask]
        y = s * r[mask]
        self.viz_ = [x,y]
        self.new_data_ = True

    def step(self):
        if(self.new_data_):
            self.ax_[0].cla()
            self.ax_[1].cla()
            self.ax_[2].cla()

            self.ax_[0].set_title('progress')
            self.ax_[0].plot(self.data_)
            self.ax_[1].set_title('histogram')
            self.ax_[1].hist(self.data_)
            print 'std', np.std(self.data_)
            print 'mean', np.mean(self.data_)
            self.ax_[2].plot(self.viz_[0], self.viz_[1], '.')
            #self.ax_[2].set_xlim(1.3 - 0.3, 1.3 + 0.3)
            #self.ax_[2].set_ylim(-0.5, 0.5)
            self.new_data_ = False
        self.fig_.canvas.draw()
        plt.pause(0.001)

    def run(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            self.step()
            rate.sleep()

def main():
    rospy.init_node('data_collection')
    node = LidarStats()
    node.run()

if __name__ == "__main__":
    main()
