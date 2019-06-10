import numpy as np
V = np.array([[3, 12], [4, 5]])
b = np.array([4, 5])
# V /=np.hypot(V[:,0],V[:,1])
print(V)
a = np.array([1, 2, 3])
b = np.array([4, 5, 6, 7])
c = max(a, b, key=len)
print(c)
a = [True, False]
b = [False, False]
c = a and b
a1 = np.array(a)
b1 = np.array(b)
print([a1, b1])
c1 = np.all([a1, b1], axis=0)
print(c1)
print(c)
cell_width = 30
cell_heigh = 30
nrow = 4
ncol = 5
# *cell_heigh/2
mesh = np.meshgrid(np.linspace(-ncol, ncol, ncol + 1),
                   np.linspace(-nrow, nrow, nrow + 1))
print(mesh)
mesh_reshap = np.reshape(mesh, (2, -1)).T
print(mesh_reshap)
zeros = np.zeros(((ncol + 1) * (nrow + 1), 1))
print(zeros)
hstack = np.hstack([mesh_reshap, zeros])
print(hstack)
objects_points = [hstack] * 20
print(objects_points)
