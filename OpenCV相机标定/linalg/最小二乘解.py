# solve可以用来解线性方程组Ax=b，也就是求解x，A是方阵
import numpy as np
from scipy import linalg

m,n = 500,50
A = np.random.rand(m,m)
B = np.random.rand(m,n)
X1 = linalg.solve(A,B)
X2 = np.dot(linalg.inv(A),B)
print(np.allclose(X1,X2))
# lstsq比solve更一般化，它不需要求矩阵A是正方形的，
# 也就是说方程个数可以少于，等于，多于未知数的个数。
# 它找到一组解想，使得||b-Ax||最小
x = np.array([0, 1])
y = np.array([-1, 0.2])
A = np.vstack([x, np.ones(len(x))]).T
print(A)
m, c = np.linalg.lstsq(A, y, rcond=None)[0]
print(m, c)
