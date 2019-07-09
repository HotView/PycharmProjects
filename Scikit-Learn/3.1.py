import numpy as np
import matplotlib.pyplot as plt
n_dot = 20
x = np.linspace(0,1,n_dot)
y = np.sqrt(x)+0.2*np.random.rand(n_dot)-0.1

def plot_ploynominal_fit(x,y,order):
    p = np.poly1d(np.polyfit(x,y,order))
    # 画出拟合出来多项式表达的曲线以及原始的点
    t = np.linspace(0,1,200)
    plt.plot(x,y,'ro',t,p(t),'-',t,np.sqrt(t),'r--')
    return p
plt.figure(figsize=(18,4))
titles = ["Under Fiting","Fitting","Over Fitting"]
models = [None,None,None]
for index,order in enumerate([1,3,10]):
    plt.subplot(1,3,index+1)
    models[index] = plot_ploynominal_fit(x,y,order)
    plt.title(titles[index],fontsize = 20)

plt.show()