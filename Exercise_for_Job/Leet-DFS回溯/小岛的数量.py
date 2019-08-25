# 遍历所有的不需要进行回溯，如果搜索所有的可能的路径，则需要进行回溯。路径需要回溯，状态不需要回溯
# 如果状态也回溯，那么就是考虑所有的可能的路径结果
def dfs(grid, x, y):
    visited[x][y] = True
    dx = [-1, 0, 1, 0]
    dy = [0, 1, 0, -1]
    for i in range(4):
        a = x + dx[i]
        b = y + dy[i]
        if (a >= 0 and a<n and b>=0 and b < m and grid[a][b] == "1" and not visited[a][b]):
            print(a,b)
            print("####")
            dfs(grid, a, b)
def solution():
    if not n:
        return 0
    count = 0
    for i in range(n):
        for j in range(m):
            if ((not visited[i][j]) and grid[i][j] == "1"):
                dfs(grid, i, j)
                count += 1
    return count
grid = []
n = int(input())
for i in range(n):
    grid.append(input())
m = len(grid[0])
visited = [[False for i in range(m)] for j in range(n)]
solution()

