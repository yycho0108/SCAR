import numpy as np
from icp import ICP
from matplotlib import pyplot as plt

myicp=ICP()

points1 = []
points2=[]
for i in range(10):
	points1.append((i,0))

for i in range(100):
	points2.append((i+10,1))

points1 =np.asarray(points1)
points2 =np.asarray(points2)

print(points2)
t, d, i, n = myicp.icp(points1, points2)

p1t = points1.dot(t[:2,:2].T) + t[:2,2]

print(t)



plt.plot(points1[:,0], points1[:,1], 'g+', label='p1')
plt.plot(points2[:,0], points2[:,1], 'r.', label='p2')
plt.plot(p1t[:,0], p1t[:,1], 'b--', label='p1t')
plt.legend()
plt.show()