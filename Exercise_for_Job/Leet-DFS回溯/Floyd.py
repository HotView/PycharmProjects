import math
inf = math.inf
def floyd():
    # 节点标号从1开始
    for k in range(1,n+1):
        for i in range(1,n+1):
            for j in range(1,n+1):
                d[i][j] = min(d[i][k]+d[k][j],d[i][j])
n,m,k = list(map(int,input().split()))
d =[[inf]*(n+1) for i in range(n+1)]
for i in range(n+1):
    d[i][i] = 0
for i in range(m):
    x,y,z = list(map(int,input().split()))
    d[x][y] = min(d[x][y],z)
floyd()
for i in range(k):
    x,y = list(map(int,input().split()))
    if d[x][y]!=inf:
        print(d[x][y])
    else:
        print("impossible")

