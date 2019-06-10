import matplotlib.pyplot as plt
import numpy as np
ax = plt.subplot(111)
t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2*np.pi*t)
line, = plt.plot(t, s, lw=2)
plt.annotate('local max',
             xy=(1,2),
             xytext=(0, 1),
             textcoords='offset points',
             ha='right',
             va='bottom'
             )
plt.ylim(-2, 2)
plt.show()