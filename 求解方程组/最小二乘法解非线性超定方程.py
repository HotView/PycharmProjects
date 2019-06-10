import numpy as np
from scipy.optimize import leastsq
from math import sqrt

def func(i):
    x,y,z = i
    return np.array((
    x**2-x*y+4,
    x**2+y**2-x*z-25,
    z**2-y*x+4,
    x**3+y**3+z**3-127.6))
root = leastsq(func,np.asarray((1,1,1)))
print(root[0])