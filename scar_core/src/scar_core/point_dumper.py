import numpy as np
from matplotlib import pyplot as plt

def R(x):
    c = np.cos(x)
    s = np.sin(x)
    r = [[c,-s],[s,c]]
    return np.asarray(r, dtype=np.float32)

class PointDumper(object):
    def __init__(self):
        pass

    @staticmethod
    def proc_frame(x,y,h,r):
        # known angular spacings
        a = np.linspace(0, 2*np.pi, 361, endpoint=True)
        c, s = np.cos(a), np.sin(a)
        dx, dy = r*c, r*s # in robot frame

        dx, dy = R(h).dot([dx,dy]) # diff in world frame

        # filter by nan/inf
        idx = np.isfinite(r)

        return (x+dx)[idx], (y+dy)[idx]

    def __call__(self, data):
        pts = [self.proc_frame(*d) for d in data]
        pts = np.concatenate(pts, axis=-1)
        return pts

def main():
    # generate "fake" data
    n = 10 # number of frames
    m = 361 # number of scans  
    x,y = np.random.uniform(-5.0, 5.0, size=(2,n))
    h  = np.random.uniform(-np.pi, np.pi, size=n)
    r = np.random.uniform(0.2, 5.0, size=(n,m))

    # apply simulated NaN to range data @ 75%
    nans = np.random.uniform(size=(n,m)) < 0.75
    r[nans] = np.nan

    # finalize data
    data = zip(x,y,h,r)

    # process + visualize points
    pd = PointDumper()
    pts = pd(data)
    plt.plot(pts[0], pts[1], '.')
    plt.show()

if __name__ == "__main__":
    main()
