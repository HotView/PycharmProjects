def down(row,col):
    for k in range(4):
        for j in range(col):
            for i in range(row-1,0,-1):
                if g[i][j]=="x" and g[i-1][j]!="x":
                    g[i][j],g[i-1][j]=g[i-1][j],g[i][j]
def dfs(num,x,y,row,col,target):
    if st[x][y] == True:
        return
    num[0] =num[0]+1
    st[x][y] =True
    dx = [-1,0,1,0]
    dy  =[0,1,0,-1]
    for i in range(4):
        a = x+dx[i]
        b = y+dy[i]
        if (a>=0 and a<row and b>=0 and b<col and not st[a][b] and  g[a][b]==target):
            dfs(num,a,b,row,col,target)
def search():
    count = []
    point = []
    for  i in range(row):
        for j in range(col):
            if g[i][j]!="x" and not st[i][j]:
                num = [0]
                dfs(num,i,j,row,col,g[i][j])
                if num[0]>3 :
                    count.append(num[0])
                    point.append([i,j])
    if not count:
        return None,None
    else:
        index = count.index(max(count))
        return count[index],point[index]
def dfs_solve(x,y,row,col,target):
    global remain
    g[x][y] ='x'
    remain-=1
    dx = [-1,0,1,0]
    dy  =[0,1,0,-1]
    for i in range(4):
        a = x+dx[i]
        b = y+dy[i]
        if (a>=0 and a<row and b>=0 and b<col and  g[a][b]==target):
            dfs_solve(a,b,row,col,target)
g = []
c = int(input())
for i in range(c):
    g.append(list(input()))
row = c
remain = 25
col = len(g[0])
st  =[[False]*col for i in range(row)]
nums,pt = search()
while(nums):
    x = pt[0]
    y = pt[1]
    dfs_solve(x,y,row,col,g[x][y])
    down(row,col)
    st = [[False] * col for i in range(row)]
    nums,pt= search()
    print(nums,pt)
    print("poch")
    for x in g:
        print(x)
print(remain)



# 5
# 00001
# 11111
# 00010
# 11020
# 11123

