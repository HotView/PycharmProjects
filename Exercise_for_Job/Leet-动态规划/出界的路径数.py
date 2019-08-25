# 记忆化搜索
# 记录然后剪枝
# 每一点表示当前点到出边界的数目
mod = 1000000007
def dp(m, n, k, x, y):
    if (f[x][y][k] != -1):
        return f[x][y][k]
    f[x][y][k] = 0
    if (not k):
        return 0;
    dx = [-1, 0, 1, 0]
    dy = [0, 1, 0, -1]
    for i in range(4):
        a = x + dx[i]
        b = dy[i] + y
        if (a < 0 or a == m or b < 0 or b == n):
            f[x][y][k] += 1
        else:
            f[x][y][k] += dp(m, n, k - 1, a, b)
        f[x][y][k] %= mod
    return f[x][y][k]

m, n, N, i, j =  list(map(int,input().split()))
f = [[[-1] * (N + 1) for i in range(n)] for j in range(m)]
print( dp(m, n, N, i, j)