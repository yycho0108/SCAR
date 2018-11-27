import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from matplotlib import pyplot as plt

class LidarStats(object):
    def __init__(self):
        self.fig_, self.ax_ = plt.subplots(1,2)
        self.new_data_ = False
        self.data_ = []

        self.scan_sub_ = rospy.Subscriber("/stable_scan",
                LaserScan, self.scan_cb)

    def scan_cb(self, msg):
        r = np.float32(msg.ranges)

        mask1 = np.isfinite(r)
        mask2 = (r >= msg.range_min)
        mask3 = (r < msg.range_max)
        a = np.logical_and
        mask  = a(mask1, a(mask2,mask3))
        print np.sum(mask)
        print mask.dtype
        
        entry = len(r[mask])
        self.data_.append(entry)
        self.new_data_ = True

    def step(self):
        if(self.new_data_):
            self.ax_[0].cla()
            self.ax_[1].cla()

            self.ax_[0].set_title('progress')
            self.ax_[0].plot(self.data_)
            self.ax_[1].set_title('histogram')
            self.ax_[1].hist(self.data_)
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
