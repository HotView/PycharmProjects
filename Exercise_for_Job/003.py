from collections import defaultdict
import math
def floyd():
    # 节点标号从1开始
    for k in range(0,n):
        for i in range(0,n):
            for j in range(0,n):
                g[i][j] = min(g[i][k]+g[k][j],g[i][j])
g = defaultdict(dict)
n = int(input())
m = int(input())
for i in range(n):
    for j in range(i):
        g[i][j] = math.inf
        g[j][i] = math.inf
for i  in range(m):
    a,b,t = list(map(int,input().split()))
    g[a][b] = t
    g[b][a] = t
queue = []
visit = [False]*n
queue.append(0)
count = 1
visit[0] = True
while(queue):
    t =queue.pop(0)
    for x in g[t].keys():
        if not visit[x] and g[t][x]!=math.inf:
            count+=1
            visit[x] = True
            queue.append(x)
if count!=n:
    print(-1)





