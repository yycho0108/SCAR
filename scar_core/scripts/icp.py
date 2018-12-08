import numpy as np
from sklearn.neighbors import NearestNeighbors
import cv2

class ICP():
    def __init__(self):
        pass

    @staticmethod
    def best_fit_transform(A, B):
        '''
        Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
        Input:
          A: Naxm numpy array of corresponding points
          B: Nbxm numpy array of corresponding points
        Returns:
          T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
          R: mxm rotation matrix
          t: mx1 translation vector
        '''

        # assert A.shape == B.shape

        # get number of dimensions
        m = A.shape[1]

        # translate points to their centroids
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - centroid_A
        BB = B - centroid_B

        # rotation matrix
        H = np.dot(AA.T, BB)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        # special reflection case
        if np.linalg.det(R) < 0:
           Vt[m-1,:] *= -1
           R = np.dot(Vt.T, U.T)

        # translation
        t = centroid_B.T - np.dot(R,centroid_A.T)

        # homogeneous transformation
        T = np.identity(m+1)
        T[:m, :m] = R
        T[:m, m] = t

        return T, R, t

    @staticmethod
    def best_fit_transform_ransac(A, B,
            sample_ratio=0.5,
            thresh=1.0,
            max_iter=100,
            min_in_ratio=0.6
            ):
        n = len(A)
        m_s = int(n * sample_ratio)
        m_i = int(n * min_in_ratio)

        # TODO : check m size

        best_err = np.inf
        best_fit = None

        for i in range(max_iter):
            idx = np.random.choice(n, m_s, replace=False)
            T, R, t = ICP.best_fit_transform(A[idx], B[idx])

            # compute pointwise error
            err_pt = (np.dot(A, R.T)  + t) - B
            err_pt = np.linalg.norm(err_pt, axis=-1)
            # get inliers + net error
            msk_pt = (err_pt < thresh)
            err = err_pt.sum()
            n_in = np.sum(msk_pt)

            if (n_in > m_i) and (err < best_err):
                # satisfy inlier requirement + good error
                best_err = err
                best_fit = (T, R, t, msk_pt)
                break
        if best_fit is None:
            return None

        msk = best_fit[-1]

        return T, R, t, msk

    @staticmethod
    def nearest_neighbor(src, dst):
        '''
        Find the nearest (Euclidean) neighbor in dst for each point in src
        Input:
            src: Naxm array of points
            dst: Nbxm array of points
        Output:
            distances: Euclidean distances of the nearest neighbor
            indices: dst indices of the nearest neighbor
        '''
        assert src.shape[1:] == dst.shape[1:]

        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(dst)
        distances, indices = neigh.kneighbors(src, return_distance=True)
        return distances.ravel(), indices.ravel()

    @staticmethod
    def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
        '''
        The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
        Input:
            A: Naxm numpy array of source mD points
            B: Nbxm numpy array of destination mD point
            init_pose: (m+1)x(m+1) homogeneous transformation
            max_iterations: exit algorithm after max_iterations
            tolerance: convergence criteria
        Output:
            T: final homogeneous transformation that maps A on to B, T.A = B
            distances: Euclidean distances (errors) of the nearest neighbor
            i: number of iterations to converge
        '''

        # assert A.shape == B.shape

        # get number of dimensions

        ndim = A.shape[1]
        nA, nB = A.shape[0], B.shape[0]
        assert(A.shape[1:] == B.shape[1:])

        # make points homogeneous, copy them to maintain the originals
        src = np.ones((nA,ndim+1), dtype=np.float32)
        dst = np.ones((nB,ndim+1), dtype=np.float32)
        
        src[:, :ndim] = A
        dst[:, :ndim] = B

        # apply the initial pose estimation
        if init_pose is not None:
            src = np.dot(init_pose, src)

        prev_error = 0

        for i in range(max_iterations):
            # find the nearest neighbors between the current source and destination points
            distances, indices = ICP.nearest_neighbor(src[:,:-1], dst[:,:-1])

            # compute the transformation between the current source and nearest destination points
            T,_,_ = ICP.best_fit_transform(src[:,:-1], dst[indices,:-1])
            # T, _, _, msk = ICP.best_fit_transform_ransac(src[:,:-1], dst[indices,:-1])

            src = np.dot(src,T.T) # right-multiply transform matrix

            # check error
            mean_error = np.mean(distances)
            if np.abs(prev_error - mean_error) < tolerance:
                break
            prev_error = mean_error

        # calculate final transformation
        T,_,_ = ICP.best_fit_transform(A, src[:,:-1])

        # alternative - refine transform
        # (mostly, compute the mask and validate truth)
        # opencv version - RANSAC
        # doesn't really do anything
        #T3, msk = cv2.estimateAffinePartial2D(
        #        A[None],
        #        src[None,:,:-1],
        #        method=cv2.FM_RANSAC,
        #        ransacReprojThreshold=0.005,
        #        maxIters=2000,
        #        confidence=0.999,
        #        refineIters=100
        #        )#, False)
        #print '{}/{}'.format(msk.sum(), msk.size)
        #print T, T3

        return T, distances, indices, i

def Rmat(x):
    c,s = np.cos(x), np.sin(x)
    R = np.float32([c,-s,s,c]).reshape(2,2)
    return R

def get_line(n, s):
    # return Nx2 line
    t = np.linspace(0, s, n)
    pt = np.stack([t, np.zeros_like(t)], axis=-1)
    h = np.random.uniform(-np.pi, np.pi)
    o = np.random.uniform(-1.0, 1.0)
    return pt.dot(Rmat(h).T) + o

def main():
    from matplotlib import pyplot as plt
    #np.random.seed(1)
    # parameters
    n_size = 100
    n_repeat = 100
    s_noise = 0.1

    pt_map = np.concatenate([get_line(n=33, s=1.0) for _ in range(3)], axis=0)

    # get geometric transformation
    # scan_{map} -> scan_{base_link}
    h = np.random.uniform(-np.pi/4, np.pi/4)
    R = Rmat(h)
    t = np.random.uniform(-1.0, 1.0, size=(1,2))
    T0 = np.eye(3)
    T0[:2,:2] = R
    T0[:2,2] = t

    print ('T (x,y,h) : ({}, {}, {})'.format(t[0,0], t[0,1], h))

    errs = []
    #n_sub  = 20
    #n_subs = np.linspace(1, len(pt_map), num=10).astype(np.int32)
    n_subs = [50]
    for n_sub in n_subs:
        print('n_sub = {}'.format(n_sub))
        err_i = []

        for _ in range(n_repeat):
            # subsample
            pt_scan = np.copy(pt_map[np.random.choice(pt_map.shape[0], size=n_sub, replace=False)])

            # apply transform
            pt_scan = pt_scan.dot(R.T) + t

            # apply noise
            pt_scan = np.random.normal(loc=pt_scan, scale=s_noise)
            T, distances, indices, iterations = ICP.icp(pt_scan, pt_map,
                    tolerance=0.000001)

            pt_scan_r = pt_scan.dot(T[:2,:2].T) + T[:2,2].reshape(1,2)

            # scan_{base_link} -> scan_{map}
            h2 = np.arctan2(T[1,0], T[1,1])
            t2 = T[:2,2]
            #print ('T_inv (x,y,h) : ({}, {}, {})'.format(t2[0], t2[1], h2))

            # compute error metric
            #print T0.dot(T)
            err_T = T0.dot(T)
            dt = err_T[:2,2]
            dh = np.arctan2(err_T[1,0], err_T[1,1])
            err_i.append(np.abs([dt[0], dt[1], dh]))
            #print('err (dx,dy,dh) : ({},{},{})'.format(dt[0],dt[1],dh))
        err_i = np.mean(err_i, axis=0)
        errs.append(err_i)

        plt.plot(pt_map[:,0], pt_map[:,1], '.', label='map')
        plt.plot(pt_scan[:,0], pt_scan[:,1], '.', label='scan')
        plt.plot(pt_scan_r[:,0], pt_scan_r[:,1], '.', label='scan-T')
        plt.legend()
        plt.show()

    #plt.plot(np.float32(n_subs) / len(pt_map), errs)
    #plt.legend(['dx','dy','dh'])
    #plt.xlabel('sample ratio')
    #plt.ylabel('error')
    #plt.title('icp characterization')
    #plt.show()

if __name__ == "__main__":
    main()
