import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from OpenCV相机标定.FitPlane import getFitplane

poses = np.load("../camera/poses.npz")["poses"]
points3dp = np.load("../camera/planepoints3d.npz")["points3dp"]
a,b,c = points3dp.shape
ones = np.ones((a,b,1))
homo_points3d =np.concatenate([points3dp,ones],axis=2)
# print(homo_points3d.shape)
# print(poses.shape)
# print(points3dp.shape)
all_points = []
for i in range(a):
    homo_point3d = homo_points3d[i].T
    #print(homo_point3d.shape)
    tranmat = poses[i]
    point3dc = np.dot(tranmat,homo_point3d).T
    all_points.append(point3dc)
    print(point3dc)
respoints = np.vstack(all_points)
#print(respoints)
fig = plt.figure('3d')
ax = fig.add_subplot(111,projection = "3d")
ax.scatter(respoints[:,0],respoints[:,1],respoints[:,2],c = 'r', marker = '^')
a,b,c = getFitplane(respoints)
np.savez("../camera/planeparams",params=[a,b,c])
#print(para)
X1,Y1 = np.meshgrid(np.linspace(0,10,50),np.linspace(-10,10,100))
Z1 =a*X1+b*Y1+c
ax.scatter(X1,Y1,Z1,c = 'g',marker = '.')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
