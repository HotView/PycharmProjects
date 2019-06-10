# 寻找最少转机（段数最少）问题
# 定义一个二维数组visit，判断坐标(x,y)是否是否访问，这里是访问过了就不能再访问了。
# BFS使用队列来实现，比如以1号顶点作为起点，将1号顶点放入队列
# 然后将与1号顶点相邻未被访问过的放入队列中，直到所有的顶点都被访问过。
#[(1, 5), (2, 5), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6), (7, 6), (7, 5), (7, 4), (6, 4), (5, 4), (4, 4), (4, 3),
#(4, 2), (4, 1), (3, 1), (2, 1), (1, 1)]
class Point:
    def __init__(self,x,y,value):
        self.x = x
        self.y = y
        self.value = value
        #self.pre_point = None
def BFS(Grapy,cur_x,cur_y,value,visited):
    ds = [0,1,0,-1,0]
    My_queue = []
    start_Point = Point(cur_x, cur_y,Grapy[cur_x][cur_y])
    My_queue.append(start_Point)
    visited[cur_x][cur_y] = 1
    while(My_queue):
        pop_point= My_queue.pop(0)
        if( pop_point.value == value):
            return pop_point
        cur_x = pop_point.x
        cur_y = pop_point.y
        for i in range(4):
            next_x = cur_x+ds[i]
            next_y = cur_y+ds[i+1]
            if next_x<0 or next_x>=row or next_y>=col or next_y<0 :
                continue
            if visited[next_x][next_y]==1 or Grapy[next_x][next_y]==1:
                continue
            next_Point = Point(next_x, next_y,Grapy[next_x][next_y])
            My_queue.append(next_Point)
            visited[next_x][next_y] = 1
            #next_Point.pre_point=pop_point
            dict_point[next_Point] = pop_point
    print("no anwser")
    return False
maze = [[ 1,1,1,1,1,1,1,1,1 ],
        [ 1,0,0,1,0,8,1,0,1 ],
        [ 1,0,0,1,1,0,0,0,1 ],
        [ 1,0,1,0,1,1,0,1,1 ],
        [ 1,0,0,0,0,1,0,0,1 ],
        [ 1,1,0,1,0,1,0,0,1 ],
        [ 1,1,0,1,0,1,0,0,1 ],
        [ 1,1,0,1,0,0,0,0,1 ],
        [ 1,1,1,1,1,1,1,1,1 ]]
dict_point = {}
row = len(maze)
col = len(maze[0])
value = 8
visited=[[0 for j in range(col)] for i in range(row)]
# while(end_point):
#     print(end_point.x,end_point.y)
#     end_point = end_point.pre_point
endPoint = BFS(maze,1,1,value,visited)
res = []
if endPoint:
    while(True):
        res.append((endPoint.x,endPoint.y))
        if endPoint in dict_point.keys():
            endPoint = dict_point[endPoint]
        else:
            break
print(res)

