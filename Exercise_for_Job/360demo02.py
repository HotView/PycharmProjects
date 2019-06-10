row = int(input())
col = int(input())
data = []
ds = [0,-1,0,1,0]
for i in range(row):
    data.append(list(map(int,input().split())))
def DFS(graph,start_x,start_y,dp_):
    if dp_[start_x][start_y] != -1:
        return dp_[start_x][start_y]
    dp_[start_x][start_y] = 1
    for i in range(4):
        #print(i)
        next_x = start_x + ds[i]
        next_y = start_y + ds[i+1]
        if next_y <0 or  next_x < 0 or next_x >= row or next_y >= col:
            continue
        if graph[next_x][next_y] <= graph[start_x][start_y]:
            continue
        dp_[start_x][start_y] = max(dp_[start_x][start_y],1+DFS(graph,next_x,next_y,dp_))
    return dp_[start_x][start_y]
dp_ = [[-1 for i in range(col)] for i in range(row)]
ans = 0
for i in range(row):
    for j in range(col):
        ans = max(ans,DFS(data,i,j,dp_))
print(max(dp_))
print(ans)