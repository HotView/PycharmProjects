from collections import defaultdict
from itertools import combinations
hashmap = defaultdict(int)
import random
t,k= list(map(int,input().split()))
data = []
for i in range(t):
    data.append(list(map(int,input().split())))
f = [0]*100000
for i in range(t):
    l = data[i][0]
    r = data[i][1]+1
    temp = r-l
    for j in range(l, r):
        if f[j]==0:
            for x in range(k, r, k):
                if j>=x :
                    temp += j-x+1
                    f[j]+=j-x+1
        else:
            temp+=f[j]
            print("haha")
    print(temp)

