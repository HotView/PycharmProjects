import numpy as np
from  scipy import optimize
import matplotlib.pyplot as plt
def func2(x, A, k, theta):
    return A * np .sin(2*np.pi*k*x+theta)
x = np.linspace(0, 2*np.pi, 100)
A, k, theta = 10, 0.34, np.pi/6 # 真实数据的函数参数
p0 = [7, 0.40, 0]
# 加入噪声之后的实验数掘
y0 = func2(x,A,k,theta)
np.random.seed(0)
yl = y0 + 2 * np.random.randn(len(x))
popt, _ = optimize.curve_fit(func2, x, yl, p0=p0)
print(popt)
plt.plot(x,y0,'r')
plt.plot(x,yl,'b')
plt.show()