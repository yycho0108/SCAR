import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt

class MapV1(object):
    """ Map Occ-Grid Representation """
    def __init__(self, w=5.0, h=5.0, res=0.02):
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
        i = np.int32(np.round((-x / self.res_)+(self.n_ /2)))
        j = np.int32(np.round((-y / self.res_)+(self.m_ /2)))
        return np.stack([i,j], axis=-1)

    def ij2xy(self, uv):
        # returns Nx2 map-coordinate points and return their physical representation
        # in the map, x is pointing up and y is pointing left
        # i.e. +x = -v, +y = -u
        i,j = np.transpose(uv)
        x = -(i-(self.n_/2))*self.res_
        y = -(j-(self.m_/2))*self.res_
        return np.stack([x,y], axis=-1)

    def update(self, p_xy, c=1.0):
        p_ij = self.xy2ij(p_xy) # Nx2
        i, j = p_ij.T
        a = np.logical_and
        # range-check mask
        mask = a(a((0<=i), (i<self.n_)),a((0<=j), (j<self.m_)))
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
        # TODO : ray-casting?
        return map_xy[dist < radius]

    def show(self, ax=None):
        mmx, mmn = self.map_.max(), self.map_.min()
        map_norm = (self.map_ - mmn) / (mmx-mmn+1e-9)
        if ax is None:
            ax = plt.gca()
        ax.imshow(map_norm, cmap='gray')

class MapV2(object):
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

    def update(self, p_xy, c=1.0):
        p_k = self.xy2k(p_xy)
        for k in p_k:
            self.map_[tuple(k)] += c

    def query(self, origin,
            radius=5.0, thresh=3.0):
        map_xy = self.k2xy(self.map_.keys())
        mask_v = np.greater_equal(self.map_.values(), thresh)
        mask_p = np.linalg.norm(map_xy - np.reshape(origin, [1,2])) < radius
        mask   = np.logical_and(mask_v, mask_p)
        return map_xy[mask]

    def show(self, ax=None):
        xy = self.k2xy(self.map_.keys())
        if ax is None:
            ax = plt.gca()
        ax.scatter(xy[:,0], xy[:,1])

def main():
    points = np.random.uniform(-2.5, 2.5, size=(5,2))
    fig, (ax0,ax1) = plt.subplots(1,2)

    map1 = MapV1()
    map1.update(points)
    map1.update(points)
    map1.update(points)
    print map1.query(origin=(0,0))
    map1.show(ax=ax0)

    map2 = MapV2()
    map2.update(points)
    map2.update(points)
    map2.update(points)
    print map2.query(origin=(0,0))
    map2.show(ax=ax1)

    plt.show()

if __name__ == "__main__":
    main()
