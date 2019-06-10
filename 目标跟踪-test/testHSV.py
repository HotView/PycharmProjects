import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
X1,Y1 = np.meshgrid(np.linspace(0,256,256),np.linspace(0,180,180))
X,Y = np.meshgrid(np.linspace(0,300,300),np.linspace(0,300,300))
roi = cv2.imread('3.png')
hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
##
target = cv2.imread('33.png')
test = cv2.imread('red.png')
hsvtest = cv2.cvtColor(test,cv2.COLOR_BGR2HSV)
hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)
print(hsvt.shape)
#计算直方图
roihist = cv2.calcHist([hsvtest],[0,1],None,[180,256],[0,180,0,256])
#np.savetxt("data.txt",roihist)
#归一化，使得值全部在0到255之间
fig3d = plt.figure()
ax = fig3d.gca(projection = '3d')
ax.plot_surface(X1,Y1,roihist,cmap='hot')
plt.show()