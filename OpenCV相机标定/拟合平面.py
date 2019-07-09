#利用最小二乘法来拟合平面方程，使得最后的结果得到一个最优的结果
import scipy.linalg
import matplotlib.pyplot as plt
import numpy as np
import cv2
x,y = np.meshgrid(np.linspace(-5,5,11),np.linspace(-5,5,11))
fig = plt.figure()
X = x.reshape(-1,1)
Y = y.reshape(-1,1)
z = 5*x+8*y+6
noise = (np.random.rand(11,11)-0.5)/2
Z1 = z+noise
Z1 = Z1.reshape(-1,1)
x0 = x.reshape(-1,1)
y0 = y.reshape(-1,1)
print(x0.shape)
one = np.ones((len(x0),1))
A = np.hstack([x0, y0,one])
print(A.shape)
a,b, c = np.linalg.lstsq(A, Z1, rcond=None)[0]
print(a,b,c)