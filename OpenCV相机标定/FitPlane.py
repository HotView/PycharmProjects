from scipy.optimize import leastsq
import numpy as np
def residuals(p,Z,Y,X):
    a,b,c = p
    return Z-(a*X+b*Y+c)
def getFitplane(points):
    """
    :param points:3-D ndarray
    :return:a,b,c
    """
    X = points[:,0]
    Y = points[:,1]
    Z = points[:,2]
    p = [1,1,1]
    r = leastsq(residuals,p,args=(Z,Y,X))
    return r[0]
# x,y = np.meshgrid(np.linspace(-5,5,11),np.linspace(-5,5,11))
# X = x.reshape(-1,1)
# Y = y.reshape(-1,1)
# z = 5*x+8*y+6
# noise = (np.random.rand(11,11)-0.5)/2
# Z = (z+noise).reshape(-1,1)
# A = np.hstack([X,Y,Z])
# print(A.shape)
# para = getFitplane(A)
# print(para)
