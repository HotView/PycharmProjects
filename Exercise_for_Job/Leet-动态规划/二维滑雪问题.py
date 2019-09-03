def dp(x,y):
    if f[x][y] != -1:
        return f[x][y]
    f[x][y] = 1
    dx = [-1,0,1,0]
    dy = [0,1,0,-1]
    for i in range(4):
        a = dx[i]+x
        b = dy[i]+y
        if(a>=0 and a<n and b>=0 and b<m and graph[a][b]>graph[x][y]):
            f[x][y] = max(f[x][y],dp(a,b)+1)
    return f[x][y]
graph = []
n,m = list(map(int,input().split()))
for i in range(n):
    graph.append(list(map(int,input().split())))
f = [[-1]*m for i in range(n)]
res = 0
for i in range(n):
    for j in range(m):
        res = max(res,dp(i,j))
print(res)