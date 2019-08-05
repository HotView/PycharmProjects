# 仅仅遍历，没有回溯
"""
#测试数据
6 4
0 1 1 0
0 1 0 0
0 1 1 1
0 1 0 1
0 1 0 1
1 1 4 3
"""
ds=  [0,1,0,-1,0]
res = False
def dfs(Graph,cur_x,cuy_y,value,visit,depth):
    if Graph[cur_x][cuy_y]==value:
        res=True
        return
    visit[cur_x][cuy_y] = 1
    print(Graph[cur_x][cuy_y],cur_x,cuy_y)
    depth = depth+1
    for i in range(4):
        next_x = cur_x+ds[i]
        next_y = cuy_y+ds[i+1]
        if next_x<0 or next_y<0 or next_x>=row or next_y>=col:
            continue
        if visit[next_x][next_y]==1 or Graph[next_x][next_y]==0:
            continue
        dfs(Graph,next_x,next_y,value,visit,depth)
maze= []
linedata = list(map(int,input().split()))
row = linedata[0]
col = linedata[1]
for i in range(row):
    maze.append(list(map(int,input().split())))
visit = [[0 for i in range(col)] for j in range(row)]
depth = 0
if __name__ == '__main__':
    dfs(maze,0,2,4,visit,depth)
