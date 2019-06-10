import numpy as np
FLANN_INDEX_KDTREE = 0
indexParams = dict(algorithm = FLANN_INDEX_KDTREE,trees = 5)
searchParams = dict(checks = 50)

print(indexParams)
print(searchParams)

a = [[[0,0] for i in range(5)] for i in range(5)]
print(a)