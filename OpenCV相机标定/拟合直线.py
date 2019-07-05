import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def f_1(x,A,B):
    return A*x+B
def f_2(x,A,B,C):
    return A*x*x+B*x+C
def f_3(x,A,B,C,D):
    return A*x*x*x+B*x*x+C*x+D
def plot_test():
    plt.figure()
    #拟合点
    x0 = [1,2,3,4,5]
    y0 = [1,3,8,18,36]
    #绘制散点
    plt.scatter(x0,y0,25,"red")
    A1,B1 = curve_fit(f_1,x0,y0)[0]
    x1 = np.arange(0,6,0.01)
    y1 = A1*x1+B1
    plt.plot(x1,y1,'g')
    P2 = curve_fit(f_2,x0,y0)[0]
    A2, B2, C2 = P2
    print(P2)
    x2 = np.arange(0,6,0.01)
    y2 = A2*x2*x2+B2*x2+C2
    plt.plot(x2,y2,'b')
    plt.show()


plot_test()