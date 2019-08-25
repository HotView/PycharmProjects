# 思想就是回溯+递归来实现。
# 路径保存是在递归进入时保存，退出时回溯即可。
#如果要搜索所有的路径，状态需要回溯。状态标记是在进入递归时调用，结束时恢复原标记即可。
# 搜索的路径可以迂回，但是不会形成环。路径上的每一个点都只经过一次。
# 设置两个数组迷宫maze，visit记录某个位置是否被访问。
"""
#测试数据
6 4
0 1 1 0
0 1 0 0
0 1 2 3
0 9 8 5
0 1 0 6
1 1 4 7
"""
dx = [-1,0,1,0]
dy = [0,1,0,-1]
def DFS(Graph,cur_x,cur_y,value):
    global corrd
    if Graph[cur_x][cur_y] == value:
        print("sucess")
        print(path)
        print(corrd)
        #path.pop()#回溯
        return
    visit[cur_x][cur_y] = 1
    path.append(Graph[cur_x][cur_y])
    for i in range(4):
        next_x = cur_x+dx[i]
        next_y = cur_y+dy[i]
        if (next_x>=0 and next_y >=0 and next_x<row and next_y<col and visit[next_x][next_y]== 0 and Graph[next_x][next_y] !=0):
            corrd.append((next_x, next_y))
            DFS(Graph,next_x,next_y,value)
            corrd.pop()
              # 回溯
    path.pop()
    visit[cur_x][cur_y] = 0

maze= []
linedata = list(map(int,input().split()))
row = linedata[0]
col = linedata[1]
for i in range(row):
    maze.append(list(map(int,input().split())))
visit = [[0 for i in range(col)] for j in range(row)]
path = []
corrd = []
value = 4
DFS(maze,2,1,value)
