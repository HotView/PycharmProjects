# 4 5 1 4
# 10 20 30 40
# 1 2 2
# 1 3 3
# 2 3 2
# 3 4 3
# 2 4 4
import math
def dijsktra():
    s= 0
    dist[s] = 0
    for i in range(n):
        # 寻找距离最短的点
        t = -1
        for j in range(1,n+1):
            if(not st[j] and (t==-1 or dist[t]>dist[j])):
                t = j
        st[t] = True
        # 更新所有点到原点的距离
        for j in range(1,n+1):
            dist[j] = min(dist[j],dist[t]+g[t][j])
n,m ,s,d =list(map(int,input().split()))
data = input()
st = [False]*(n+2)
dist = [math.inf]*(n+2)
g = [[math.inf]*(n+2) for i in range(n+2)]
for i in range(m):
    x,y,z = list(map(int,input().split()))
    g[x][y] = g[y][x] = min(g[x][y],z)
dijsktra()
print(dist)

