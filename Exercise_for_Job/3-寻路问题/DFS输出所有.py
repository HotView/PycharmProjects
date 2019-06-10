# 思想就是回溯+递归来实现。
# 这里是两个回溯的问题。
# 设置两个数组迷宫maze，visit记录某个位置是否被访问。
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
ds = [0,1,0,-1,0]
def DFS(Graph,cur_x,cur_y,value,visit,path):
    path.append(Graph[cur_x][cur_y])
    if Graph[cur_x][cur_y] == value:
        print(path)
        path.pop()#回溯
    else:
        visit[cur_x][cur_y] = 1
        for i in range(4):
            next_x = cur_x+ds[i]
            next_y = cur_y+ds[i+1]
            if next_x<0 or next_y <0 or next_x>=row or next_y>=col:
                continue
            if visit[next_x][next_y]== 1 or Graph[next_x][next_y] == 0:
                continue
            DFS(Graph,next_x,next_y,value,visit,path)
        path.pop()#回溯
maze= []
linedata = list(map(int,input().split()))
row = linedata[0]
col = linedata[1]
for i in range(row):
    maze.append(list(map(int,input().split())))
visit = [[0 for i in range(col)] for j in range(row)]
path = []
value = 4
DFS(maze,0,2,value,visit,path)
