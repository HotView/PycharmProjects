import copy
t= int(input())
all_n = []
all_m = []
all_data = []
all_startx = []
all_starty = []
all_endx = []
all_endy = []
#copy.deepcopy()
def read_data():
    global t
    global all_n
    global all_m
    global all_startx
    global all_starty
    global all_endx
    global all_endy
    for i in range(t):
        n, m = list(map(int, input().split()))
        all_n.append(n)
        all_m.append(m)
        data = []
        for j in range(n):
            line = input()
            temp_line = [1 for i in range(m)]
            for i in range(m):
                if line[i] == '.':
                    temp_line[i] = 2
                else:
                    temp_line[i] = 1
            data.append(temp_line)
        all_data.append(data)
        startx, starty = list(map(int, input().split()))
        all_startx.append(startx)
        all_starty.append(starty)
        endx, endy = list(map(int, input().split()))
        all_endx.append(endx)
        all_endy.append(endy)
def dfs(data,n,m,curx,cury,endx,endy,cishu):
    if(curx==endx and cury==endy):
        global success
        success = success+1
        return
    dx = [-1,0,1,0]
    dy = [0,1,0,-1]
    data[curx][cury]-=1
    for i in range(4):
        a= curx+dx[i]
        b = cury+dy[i]
        if(a>=0 and a<n and b>=0 and b<m and data[a][b]>0):
            dfs(data,n,m,a,b,endx,endy,cishu)
read_data()
for i in range(t):
    n = all_n[i]
    m = all_m[i]
    success = 0
    data = all_data[i]
    startx = all_startx[i]-1
    starty = all_starty[i]-1
    endx = all_endx[i]-1
    endy = all_endy[i]-1
    cishu = data[endx][endy]
    dfs(data,n,m,startx,starty,endx,endy,cishu)
    if success<cishu:
        print("NO")
    else:
        print("YES")

