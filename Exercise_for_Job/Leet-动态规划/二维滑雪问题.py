def dp(x,y):
    if f[x][y] != -1:
        return f[x][y]
    f[x][y] = 1
    dx = [-1,0,1,0]
    dy = [0,1,0,-1]
    for i in range(4):
        a = dx+x
        b = dy+y
        if(a>=0 and a<n and b>=0 and b<m ):
            f[x][y] = max(f[x][y],dp(a,b)+1)
    return f[x][y]
graph = [[]]
n = len(graph)
m = len(graph[0])
f = [[-1]*m for i in range(n)]
res = 0
for i in range(n):
    for j in range(m):
        res = max(res,dp(i,j))
print(res)