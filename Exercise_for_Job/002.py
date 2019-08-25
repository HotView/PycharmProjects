from collections import defaultdict
hashmap = defaultdict
def dfs(x,y,target):
    g[x][y]=6
    dx = [-1,0,1,0]
    dy = [0,1,0,-1]
    for i in range(4):
        a = x+dx[i]
        b = y+dy[i]
        if(a>=0 and a<5 and b>=0 and b<5 and g[a][b]==target ):
            dfs(a,b,target)
t = int(input())
data = []
alln = []
allm = []
for i in range(t):
    n,m = list(map(int,input().split()))
    alln.append(n)
    allm.append(m)
    g = []
    for j in range(n):
        g.append(input())
    data.append(g)
for i in range(t):
    g = data[i]
    start = []
    for i in range(alln[i]):
        for j in range(allm[i]):
            if g[i][j] == 'S':
                start = [i,j]
    dx = [-1, 0, 1, 0]
    dy = [0, 1, 0, -1]
    for i in range(4):
        a = start[0]+dx[i]
        b = start[1]+dy[j]
        if a<0:
            a= n-1
        if b<0:
            b =m-1
        if a>n-1:
            a = n-1
        if b>m-1:
            b = m-1
        if g[a][b]!="#":
            print("Yes")
            break
    else:
        print("No")






