# 4 5 1 4
# 10 20 30 40
# 1 2 2
# 1 3 3
# 2 3 2
# 3 4 3
# 2 4 4
import math
from collections import defaultdict
def dijsktra():
    dist[s] = 0
    for i in range(n):
        t = -1
        # 寻找最短距离
        for j in range(1,n+1):
            if(not st[j] and (t==-1 or dist[t]>dist[j])):
                t = j
        st[t] = True
        # 更新所有点到原点的距离
        for j in range(1,n+1):
            if(dist[j]>dist[t]+g[t][j]):
                dist[j] = dist[t]+g[t][j]
                prev[j] = t
            #增加添加路径的功能，寻找最短的路径下另一个最值标准
            elif(dist[j]==dist[t]+g[t][j]):
                if cake[prev[j]]<cake[t]:
                    prev[j] = t
n,m,s,d  =list(map(int,input().split()))
st = [False]*(n+2)
dist = [math.inf]*(n+2)
prev =[-1]*(n+2)
g = [[math.inf]*(n+2) for i in range(n+2)]

cake = list(map(int,input().split()))
for i in range(m):
    x,y,z = list(map(int,input().split()))
    g[x][y] = z
dijsktra()
path = []
path.append(d)
print(prev)
now = prev[d]
while(1):
    path.append(now)
    if(now==s):
        break
    now = prev[now]
values = 0
for x in path:
    values+=cake[x-1]
print(dist[n],values)
