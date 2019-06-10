#思想就是回溯+递归来实现，回溯主要是用来记录路径。
#设置两个数组迷宫maze，visit记录某个位置是否被访问
# 可以输出路径
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
def DFS(graph,cur_x,cur_y,end_value,visit,path):
    #最终的子问题遇到的情况
    if graph[cur_x][cur_y] == end_value:
        path.append([graph[cur_x][cur_y],cur_x,cur_y])
        return True
    visit[cur_x][cur_y] = 1
    path.append([graph[cur_x][cur_y],cur_x,cur_y])
    for i in range(4):
        next_x = cur_x+ds[i]
        next_y = cur_y+ds[i+1]
        if next_x<0 or next_y<0 or next_x>=row or next_y>=col:
            continue
        if graph[next_x][next_y] == 0 or visit[next_x][next_y]==1:
            continue
        #如果只是输出一条路径的话，可以找到一条后就直接返回，不要在尝试邻域的其他点了
        if DFS(graph,next_x,next_y,end_value,visit,path)==True:
            return True
    path.pop()#回溯
    return False
linedata = list(map(int, input().split()))
row = linedata[0]
col = linedata[1]
maze = []
for i in range(row):
    maze.append(list(map(int,input().split())))
visit = [[0 for i in range(col)] for j in range(row)]
# print(maze)
# print(visit)
Path = []
if DFS(maze,0,2,4,visit,Path):
    print(Path)

