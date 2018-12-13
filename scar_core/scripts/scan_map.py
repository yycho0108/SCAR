import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
import cv2
from sklearn.neighbors import NearestNeighbors

def to_polar(pts, o):
    dxy = pts - np.reshape(o, [1,2])
    r = np.linalg.norm(dxy, axis=-1)
    h = np.arctan2(dxy[:,1], dxy[:,0])
    return np.stack([r,h], axis=-1)

#def to_polar(pts, o):
#    dxy = pts - np.reshape(o, [1,2])
#    r = np.linalg.norm(dxy, axis=-1)
#    h = np.arctan2(dxy[:,1], dxy[:,0])
#    return np.stack([r,h], axis=-1)

class DenseMap(object):
    """ Map Occ-Grid Representation """
    def __init__(self, w=10.0, h=10.0, res=0.02):
        self.w_ = w
        self.h_ = h
        self.res_ = res
        self.n_ = n = int(np.ceil(h / res))
        self.m_ = m = int(np.ceil(w / res))
        self.map_ = np.zeros((n,m), dtype=np.float32)
        self.pts_ = None # TODO : not used right now, enable cachcing point lookups at some point

    def xy2ij(self, xy):
        # returns Nx2 physical points and return their map-coordinate representation
        # in the map, x is pointing up and y is pointing left
        # i.e. +x = -v, +y = -u
        x, y = xy.T
        i = np.int32(np.round((-y / self.res_)+(self.n_ /2)))
        j = np.int32(np.round(( x / self.res_)+(self.m_ /2)))
        return np.stack([i,j], axis=-1)

    def ij2xy(self, ij):
        # returns Nx2 map-coordinate points and return their physical representation
        # in the map, x is pointing up and y is pointing left
        # i.e. +x = -v, +y = -u
        i,j = np.transpose(ij)
        y = -(i-(self.n_/2))*self.res_
        x = (j-(self.m_/2))*self.res_
        return np.stack([x,y], axis=-1)

    def update(self, p_xy, c=1.0, origin=None):
        p_ij = self.xy2ij(p_xy) # Nx2
        i, j = p_ij.T
        mask = np.logical_and.reduce([0<=i, i<self.n_, 0<=j, j<self.m_])
        # range-check mask
        # log range check failures
        #if not np.all(mask):
        #    print('Some of the passed points were not plottable in the map!')
        #    print('Either increase the map size or configure the map to be dynamic.')

        i, j = i[mask], j[mask]
        self.map_[i,j] += c

    def query(self, origin,
            radius=5.0,
            thresh=3.0):
        """ all points that are visible from the origin """
        # TODO : efficiently query the map by limiting the field here
        map_ij = np.stack(np.where(self.map_ >= thresh), axis=-1)
        if map_ij.size <= 0:
            # No points should be visible within the map
            return []
        map_xy = self.ij2xy(map_ij)
        dist = np.linalg.norm(map_xy - np.expand_dims(origin, 0), axis=-1)
        print np.sum( dist < radius ), ' ?' 
        # TODO : ray-casting?
        return map_xy[dist < radius]

    def show(self, ax=None):
        # dilate just for visualization

        map_norm = np.float32(self.map_ > 0)
        #map_norm = cv2.dilate(self.map_, np.ones((3,3), dtype=np.uint8))
        #map_norm = np.float32(map_norm > 0)

        # show
        if ax is None:
            ax = plt.gca()
        ax.imshow(map_norm, cmap='gray')

class SparseMap(object):
    """ Map Dict Representation """
    def __init__(self, res=0.01):
        # self.map_[(x,y)] = count!
        self.map_ = defaultdict(lambda:0.0)
        self.res_ = res

    def xy2k(self, xy):
        #Nx2 xy --> Nx1 (k)
        k = np.int32(np.round(xy/self.res_))
        return [tuple(e) for e in k]

    def k2xy(self, k):
        # Nx1 (k) --> Nx2 (xy)
        k = np.asarray(k, dtype=np.float32) # Nx2
        return k * self.res_

    def update(self, p_xy, c=1.0, origin=None):
        # 1. clear
        map_xy = self.query(origin, thresh=0.0)
        if len(map_xy) > 0:
            p_rh   = to_polar(p_xy,   origin) #N,2
            map_rh = to_polar(map_xy, origin) #N,2

            neigh = NearestNeighbors(n_neighbors=1)
            neigh.fit(p_rh[:,1:])
            distances, indices = neigh.kneighbors(map_rh[:,1:],
                    return_distance=True)
            distances = distances[:,0]
            indices = indices[:,0]

            rmask = p_rh[indices,0] > map_rh[:,0]
            hmask = (distances < np.deg2rad(1.0))
            mask = np.logical_and(rmask, hmask)

            for k in self.xy2k(map_xy[mask]):
                self.map_[tuple(k)] -= 1.0

        # 2. add
        p_k = self.xy2k(p_xy)
        for k in p_k:
            self.map_[tuple(k)] += c

    def query(self, origin,
            radius=5.0, thresh=3.0):
        map_xy = self.k2xy(self.map_.keys())
        if len(map_xy) <= 0:
            return []
        mask_v = np.greater_equal(self.map_.values(), thresh)
        mask_p = np.linalg.norm(map_xy - np.reshape(origin, [1,2]), axis=-1) < radius
        mask   = np.logical_and(mask_v, mask_p)
        return map_xy[mask]

    def show(self, ax=None):
        xy = self.k2xy(self.map_.keys())
        if ax is None:
            ax = plt.gca()
        ax.scatter(xy[:,0], xy[:,1])

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

class ProbMap(DenseMap):
    def __init__(self, w=10.0, h=10.0, res=0.02):
        super(Prob, self).__init__(w,h,res)

        # log-odds parametrization
        self.l0_ = 0.0
        self.locc_ = 0.4
        self.lfree_ = -0.4

        self.beam_i_ = (1.0 / (2*np.pi)) # beam interval
        self.beam_w_ = 2.3e-3

        # From Lidar Specs:
        #-Pulse Rep Frequency 1.8 kHz
        #-Pulse Duration 200 usec
        #-Peak Power 2.1 mW
        #-Beam Diameter 2.3 mm
        #-Beam Divergence -1.5 mrad

    def inverse_sensor_model(self,
            origin, scans,
            map_occ, map_loc,
            beam_intbeam_div
            ):
        map_rh = to_polar_h(origin, map_loc)

        # beam character mask
        bmask = (map_rh + np.pi) % (1.0 / (2*np.pi)) - np.pi
        bmask = np.abs(bmask) < self.beam_w_
        # TODO : implement
        #map_loc - 

    def update(self, p_xy,
            c=1.0, origin=None):
        # 1. clear

        # TODO : avoid hard-coding 5.0 here
        # pass in more parameters at __init__()
        ox, oy, oh = origin
        dmask = np.linalg.norm(self.grid_ - [[[ox, oy]]], axis=-1) < 5.0
        self.map_[dmask] += self.inverse_sensor_model(origin, p_xy,
                self.map_[dmask],
                self.grid_[dmask]
                )

        map_xy = self.query(origin)
        p_rh   = to_polar(p_xy,   origin) #N,2
        map_rh = to_polar(map_xy, origin) #N,2

        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(p_rh[:,1:])
        distances, indices = neigh.kneighbors(map_rh[:,1:],
                return_distance=True)

        rmask = p_rh[indices,0] > map_rh[:,0]
        hmask = (distances < np.deg2rad(1.0))
        mask = np.logical_and(rmask, hmask)

        self.map_[self.xy2ij(map_xy[mask])] = 0.0

        # 2. add
        #for i, (r,h) in enumerate(map_rh):
        #    if r > 


    def query(self, origin, radius=5.0, thresh=0.75):
        # >75% chance considered occupied
        pass

def main():
    points = np.random.uniform(-2.5, 2.5, size=(10,2))
    fig, (ax0,ax1) = plt.subplots(1,2)

    map1 = DenseMap()

    print '!!'
    print points
    print map1.ij2xy(map1.xy2ij( points ))
    print '=='

    map1.update(points, origin=(0,0))
    map1.update(points, origin=(0,0))
    map1.update(points, origin=(0,0))
    print 'q1', map1.query(origin=(0,0))
    map1.show(ax=ax0)

    map2 = SparseMap()
    map2.update(points, origin=(0,0))
    map2.update(points, origin=(0,0))
    map2.update(points, origin=(0,0))
    print 'q2', map2.query(origin=(0,0))
    map2.show(ax=ax1)

    plt.show()

if __name__ == "__main__":
    main()
