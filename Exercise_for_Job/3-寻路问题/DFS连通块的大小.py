
def dfs(x,y,row,col):
    global num
    if st[x][y] == True:
        return
    num =num+1
    st[x][y] =True
    dx = [-1,0,1,0]
    dy  =[0,1,0,-1]
    for i in range(4):
        a = x+dx[i]
        b = y+dy[i]
        if (a>=0 and a<row and b>=0 and b<col and not st[a][b] and  g[a][b]=="0"):
            dfs(a,b,row,col)
g = []
c = int(input())
for i in range(c):
    g.append(input())
row = c
col = len(g[0])
point = []
st  =[[False]*col for i in range(row)]
for  i in range(row):
    for j in range(col):
        if g[i][j]=="0" and not st[i][j]:
            num = 0
            dfs(i,j,row,col)
            if num>3:
                print(num)
                point.append([i,j])
print(point)
