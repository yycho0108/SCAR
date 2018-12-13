import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
import cv2
from sklearn.neighbors import NearestNeighbors

def R2(x):
    c = np.cos(x)
    s = np.sin(x)
    return np.reshape([c,-s,s,c], (2,2))

def anorm(x):
    return (x + np.pi) % (2*np.pi) - np.pi

def to_polar(pts, o):
    dxy = pts - np.reshape(o[:2], [1,2])
    r = np.linalg.norm(dxy, axis=-1)
    h = np.arctan2(dxy[:,1], dxy[:,0])
    return np.stack([r,h], axis=-1)

def to_polar_h(pts, o):
    # points w.r.t origin accounting heading
    # pts' = (pts - o).dot(R(oh)) # inverse rotation

    op = np.reshape(o[:2], (1,2))
    oh = o[-1]

    pts = (pts - op).dot(R2(oh))

    r = np.linalg.norm(pts, axis=-1)
    h = np.arctan2(pts[:,1], pts[:,0])

    #pts = (pts - op)
    #r = np.linalg.norm(pts, axis=-1)
    #h = np.arctan2(pts[:,1], pts[:,0]) - oh
    #h = anorm(h)

    return np.stack([r,h], axis=-1)

class DenseMap(object):
    """ Map Occ-Grid Representation """
    def __init__(self, w=10.0, h=10.0, res=0.02):
        self.w_ = w
        self.h_ = h
        self.res_ = res
        self.n_ = n = int(np.ceil(h / res))
        self.m_ = m = int(np.ceil(w / res))
        self.o_ = ((self.n_+1) / 2.0, (self.m_+1) / 2.0)
        self.map_ = np.zeros((n,m), dtype=np.float32)
        self.pts_ = None # TODO : not used right now, enable cachcing point lookups at some point

    def xy2ij(self, xy):
        # returns Nx2 physical points and return their map-coordinate representation
        # in the map, x is pointing up and y is pointing left
        # i.e. +x = -v, +y = -u
        x, y = xy[...,0], xy[...,1]
        i = np.int32(np.round(-y/self.res_+self.o_[0]-0.5))
        j = np.int32(np.round( x/self.res_+self.o_[1]-0.5))
        return np.stack([i,j], axis=-1)

    def ij2xy(self, ij):
        # returns Nx2 map-coordinate points and return their physical representation
        # in the map, x is pointing up and y is pointing left
        # i.e. +x = -v, +y = -u
        i, j = ij[...,0], ij[...,1] #np.transpose(ij)
        y = -(i+0.5-self.o_[0])*self.res_
        x = (j+0.5-self.o_[1])*self.res_
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
            return np.empty(shape=(0,2), dtype=np.float32)
        map_xy = self.ij2xy(map_ij)
        dist = np.linalg.norm(map_xy - np.reshape(origin[:2], [1,2]), axis=-1)
        #print np.sum( dist < radius ), ' ?' 
        # TODO : ray-casting?
        return map_xy[dist < radius]

    def show(self, ax=None):
        # dilate just for visualization

        map_viz = cv2.dilate(self.map_, np.ones((3,3), dtype=np.uint8))
        map_viz = np.float32(map_viz > 0)
        # show
        if ax is None:
            ax = plt.gca()
        ax.imshow(map_viz, cmap='gray')

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
        mask_p = np.linalg.norm(map_xy - np.reshape(origin[:2], [1,2]), axis=-1) < radius
        mask   = np.logical_and(mask_v, mask_p)
        return map_xy[mask]

    def show(self, ax=None):
        xy = self.k2xy(self.map_.keys())
        if ax is None:
            ax = plt.gca()
        ax.scatter(xy[:,0], xy[:,1])
        ax.set_aspect('equal', 'datalim')

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
        super(ProbMap, self).__init__(w, h, res)
        # NOTE:
        # fortunately, self.map_ is initialized to 0
        # which happens to be self.l0_

        # cache map centroid
        i, j = np.mgrid[0:self.n_, 0:self.m_]
        ij = np.stack([i,j], axis=-1)
        self.ctr_ = self.ij2xy(ij)

        # log-odds parametrization
        self.l0_ = 0.0
        self.locc_ = 0.4
        self.lfree_ = -0.4

        self.beam_i_ = (1.0 / (2*np.pi)) # beam interval
        self.beam_n_ = 361 # number of beams - TODO:hardcoded
        self.beam_w_ = 2.3e-3
        self.beam_d_ = 1.5e-3#(-1.5e-3)
        #self.beam_s_ =  beam distance standard deviation:
        # for beam_s, refer to https://www.diva-portal.org/smash/get/diva2:995686/FULLTEXT01.pdf

        # From Lidar Specs:
        #-Pulse Rep Frequency 1.8 kHz
        #-Pulse Duration 200 usec
        #-Peak Power 2.1 mW
        #-Beam Diameter 2.3 mm
        #-Beam Divergence -1.5 mrad

    def rh2bin(self, rh):
        h = (rh[:, 1]) % (2 * np.pi) # mod in python always returns positive
        bid = np.round(h / self.beam_i_).astype(np.int32)
        return bid

    def inverse_sensor_model(self,
            origin, scans,
            map_occ, map_loc, dmask
            ):
        # convert to polar coordinates w.r.t. base_link
        map_rh  = to_polar_h(map_loc, origin)
        scan_rh = to_polar_h(scans, origin)
        scan_rh = scan_rh[ np.argsort(-scan_rh[:,0]) ]
        # process longer range scans first
        # (=overrides longer range scans)

        s_bid = self.rh2bin(scan_rh)
        m_bid = self.rh2bin(map_rh)

        # store bin-wise range data
        bsr = np.full(self.beam_n_, -np.inf, dtype=np.float32)
        bsr[s_bid] = scan_rh[:,0]

        mfree = (map_rh[:,0] < bsr[m_bid]) # cells to clear
        mocc  = np.abs(map_rh[:,0] - bsr[m_bid]) < (self.res_/2.0) # cells to mark
        mfree[mocc] = False # avoid clearing marking cells

        # beam character mask
        # cells outside of beam character area should not be marked
        # considers cell+beam radius.
        c_r  = self.res_/2.0 # cell radius leeway
        bang  = map_rh[:,1] - (np.round(map_rh[:,1] / self.beam_i_) * self.beam_i_) # min deviation from nearest beam
        brad  = np.abs(self.beam_w_ + self.beam_d_ * map_rh[:,0]) # expected beam radius @ distance
        #brad  = self.beam_w_
        bmask = np.abs(map_rh[:,0] * bang) < (c_r + brad)

        # hit/miss
        map_occ[mfree & bmask] += self.lfree_
        map_occ[mocc  & bmask] += self.locc_
        self.map_[dmask] = map_occ # TODO: why should it be set like this?

    def update(self, p_xy,
            c=1.0, origin=None):
        # 1. clear
        # TODO : avoid hard-coding 5.0 here
        # pass in more parameters at __init__()
        ox, oy, oh = origin
        dmask = np.linalg.norm(self.ctr_ - [[[ox, oy]]], axis=-1) < 5.0
        self.inverse_sensor_model(origin, p_xy,
                self.map_[dmask],
                self.ctr_[dmask],
                dmask
                )

    def query(self, origin, radius=5.0, thresh=0.75):
        # >75% chance considered occupied
        # thresh input = probability, convert to log odds
        # thresh = p/(p+1) == thres
        # thresh_log = log(p/(1-p))
        thresh = np.log( thresh / (1.0 - thresh) )
        return super(ProbMap, self).query(origin, radius, thresh)

    def show(self, ax=None):
        # dilate just for visualization

        p = np.exp(self.map_)
        p = p / (1 + p)
        map_viz = (1.0 - p) # invert so that it looks like ROS visualizations

        #map_viz = np.full_like(self.map_, 0.5)
        #map_viz[ self.map_ > 0.0 ] = 0.0 # occupied
        #map_viz[ self.map_ < 0.0 ] = 1.0 # occupied

        #k = cv2.getGaussianKernel(3,1)
        #map_viz = cv2.dilate(map_viz, k)
        # show
        if ax is None:
            ax = plt.gca()
        ax.imshow(map_viz, cmap='gray')

def ordered_pt(x):
    return x[np.argsort(np.linalg.norm(x,axis=-1))]

def main():
    origin = (1.0,0.5,1.2)
    points = np.random.uniform(-2.5, 2.5, size=(20,2))
    q0 = points
    print 'q0', ordered_pt(points)

    ## test ##
    map1 = DenseMap(res=0.02)
    map1.update(points, origin=origin)
    map1.update(points, origin=origin)
    map1.update(points, origin=origin)
    q1 = map1.query(origin=origin)
    print 'q1', ordered_pt(q1)

    map2 = ProbMap(res=0.02)
    map2.update(points, origin=origin)
    map2.update(points, origin=origin)
    map2.update(points, origin=origin)
    q2 = map2.query(origin=origin)
    print 'q2', ordered_pt(q2)

    map3 = SparseMap(res=0.02)
    map3.update(points, origin=origin)
    map3.update(points, origin=origin)
    map3.update(points, origin=origin)
    q3 = map3.query(origin=origin)
    print 'q3', ordered_pt(q3)

    ## visualization ##
    fig = plt.figure()
    ax1 = fig.add_subplot(2,4,1)
    ax2 = fig.add_subplot(2,4,2)
    ax3 = fig.add_subplot(2,4,5)
    ax4 = fig.add_subplot(2,4,6)
    ax5 = fig.add_subplot(1,2,2)

    ax1.plot(q0[:,0], q0[:,1], '.')
    ax1.set_title('Source')
    map1.show(ax=ax2)
    ax2.set_title('Dense')
    map2.show(ax=ax3)
    ax3.set_title('Prob')
    map3.show(ax=ax4)
    ax4.set_title('Sparse')

    ax5.plot(q0[:,0], q0[:,1], '+', label='source', alpha=0.5) 
    ax5.plot(q1[:,0], q1[:,1], 'x', label='dense', alpha=0.5) 
    ax5.plot(q2[:,0], q2[:,1], '*', label='prob', alpha=0.5) 
    ax5.plot(q3[:,0], q3[:,1], '.', label='sparse', alpha=0.5) 
    ax5.set_title('Query Overlay')
    ax5.legend()

    plt.show()

if __name__ == "__main__":
    main()
