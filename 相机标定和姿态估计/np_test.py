import numpy as np

a= np.random.randint(5,55,(4,4))
b = np.random.randint(6,45,(4,4))
c = np.where((a>b))
print(a)
print(b)
print(c)