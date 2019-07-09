import numpy as np
from  scipy import optimize

X = np.array([8.19, 2.72 , 6.39 , 8.71 , 4.7 , 2.66, 3.78])
Y= np.array([ 7.01, 2.78,6.47, 6.71, 4.1 , 4.23, 4.05])
def residuals(p):
    k,b = p
    return Y-(k*X+b)
res = optimize.leastsq(residuals,[1,0])
k,b = res[0]
print(k,b)
"""###################################"""
# 对于一维曲线的拟合，可以使用cure_fit函数，下面使用此函数对正弦函数数据进行拟合
# 它的目标函数与leastsq不同，各个待优化参数直接作为函数的参数传入
