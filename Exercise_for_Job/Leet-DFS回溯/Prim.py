import math
def prim():
    res = 0
    for i in range(n):
        t = -1
        # 寻找到集合最近的点
        for j in range(1,n+1):
            #如果没有更新t或者t的节点大于j的节点，更新t
            if(not st[j] and(t==-1 or (dist[t]>dist[j]))):
                t=j
        if (i and dist[t]==math.inf):
            return math.inf
        if (i):
            res += dist[t]
        # 更新到集合最近的点,就等于取到加入的点距离和远距离的最小值
        for j in range(1,n+1):
            dist[j] = min(dist[j],g[t][j])
        st[t] = True
    return res
n,m = list(map(int,input().split()))
inf = math.inf
g= [[inf]*(n+2) for i in range(n+2)]
dist = [inf]*(n+2)
st = [False]*(n+2)
for i in range(m):
    x,y,z = list(map(int,input().split()))
    g[x][y] = g[y][x] = min(g[x][y],z)
res = prim()
if res!=math.inf:
    print(res)
else:
    print("impossible")



