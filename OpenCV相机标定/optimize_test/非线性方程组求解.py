from math import sin,cos
from scipy import optimize

def f(x):
    """
    return 返回的是非线性的齐次方程组
    :param x:
    :return:
    """
    x0,x1,x2 = x.tolist()
    return[ 5*x1+3,
            4*x0*x0 - 2*sin(x1*x2),
            x1*x2 - 1.5]
result = optimize.fsolve(f,[1,1,1])
print(result)
print(f(result))