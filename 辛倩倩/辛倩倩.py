import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

data = np.loadtxt("tiffdf_f.txt",skiprows=(1)).T
conv = np.corrcoef(data)
newconv = np.zeros((conv.shape))
newconv_res = np.zeros((conv.shape))
points = []

for i in range(len(conv)):
    newconv[i,i:] = sorted(conv[i,i:],reverse=True)
for m in range(len(conv)):
    for n in range(m):
        points.append(conv[m,n])
for i in range(len(conv)):
     newconv_res[i,i] = 1
print(len(points))
sortpoits = sorted(points,reverse=True)
sortindex = np.argsort(conv)
print(sortindex)
for m in range(len(conv)):
    for n in range(m):
        print(m, n)
        newconv_res[n,m] = sortpoits[0]
        newconv_res[m,n] = sortpoits[0]
        sortpoits.pop(0)
print(data.shape)
print(conv.shape)
plt.figure(figsize=(8,8),dpi=144)
plt.title("Correlation of neuron-neuron activity")
ax = plt.gca()
ax.set_xlabel("Neuron#")
ax.set_ylabel("Neuron#")
#im = ax.imshow(newconv_res,cmap='jet')
im = ax.imshow(newconv_res,cmap='jet')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
plt.show()
