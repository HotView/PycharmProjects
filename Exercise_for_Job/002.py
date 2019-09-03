from collections import defaultdict
from itertools import combinations
hashmap = defaultdict(int)
import random
t,k= list(map(int,input().split()))
data = []
for i in range(t):
    data.append(list(map(int,input().split())))
f = [0]*100000

for i in range(100000):
    f[i] = f[i-1]+f[i-2]
for i in range(t):
    l = data[i][0]
    r = data[i][1]+1
    temp = r-l
    for x in range(k, r, k):
        for j in range(l, r):
            if j>=x :
                temp += j-x+1
    print(temp)

