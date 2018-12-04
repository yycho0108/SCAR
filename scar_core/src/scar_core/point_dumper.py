import numpy as np
import time
from matplotlib import pyplot as plt

def R(x):
    c = np.cos(x)
    s = np.sin(x)
    r = [[c,-s],[s,c]]
    return np.asarray(r, dtype=np.float32)

class PointDumper(object):
    def __init__(self, viz=False):
        self.viz_ = viz
        if self.viz_:
            self.fig_, self.ax_ = plt.subplots(1,1)

        

    def proc_frame(self, x,y,h,r):
        # known angular spacings
        n = len(r)
        a = np.linspace(0, 2*np.pi, n, endpoint=True)
        c, s = np.cos(a), np.sin(a)
        dx, dy = r*c, r*s # in robot frame

        dx, dy = R(h).dot([dx,dy]) # diff in world frame

        # filter by nan/inf
        idx = np.isfinite(r)

        px, py = (x+dx)[idx], (y+dy)[idx]
        if self.viz_:
            self.visualize(px, py)
        return px, py

    def visualize(self, x, y, clear=False, draw=False, label=''):
        if clear:
            self.ax_.cla()
        #plt.ion() # will work maybe?
        self.ax_.plot(x,y, '.')
        #self.ax_.draw()

        if draw:
            self.fig_.canvas.draw()
            self.ax_.legend()
        plt.pause(0.001)
        #plt.draw()
        #plt.show()

    def __call__(self, data):
        pts = [self.proc_frame(*d) for d in data]
        pts = np.concatenate(pts, axis=-1)
        return pts

def main():
    # generate "fake" data
    n = 100 # number of frames
    m = 360 # number of scans  
    x,y = np.random.uniform(-5.0, 5.0, size=(2,n))
    h  = np.random.uniform(-np.pi, np.pi, size=n)
    r = np.random.uniform(0.2, 5.0, size=(n,m))

    # apply simulated NaN to range data @ 75%
    nans = np.random.uniform(size=(n,m)) < 0.75
    r[nans] = np.nan

    # finalize data
    data = zip(x,y,h,r)

    # process + visualize points
    pd = PointDumper(viz=True)
    pts = pd(data)
    #plt.plot(pts[0], pts[1], '.')
    plt.show()

if __name__ == "__main__":
    main()
