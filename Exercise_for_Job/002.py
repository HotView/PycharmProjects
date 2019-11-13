from collections import defaultdict
from itertools import combinations
m,n = list(map(int,input().split()))
data = []
for i in range(m):
    temp = list(map(int,input().split()))
    data.append(temp)
dx = [0,1,0,-1]
dy = [1,0,-1,0]
pos = 0
used = [[0]*n for i in range(m)]
res = []
x = 0
y = 0
for i in range(m*n):
    res.append(data[x][y])
    used[x][y] = 1
    a = dx[pos] + x
    b = dy[pos] + y
    if a<0 or b<0 or a>=m or b>=n or used[a][b]:
        pos=(pos+1)%4
        x = dx[pos] + x
        y = dy[pos] + y
    else:
        x = a
        y = b
last = m*n-1
for i in range(m*n):
    if i == last:
        print(res[i])
    else:
        print(res[i],end=',')
