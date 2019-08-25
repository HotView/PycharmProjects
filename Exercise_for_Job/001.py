from collections import defaultdict
hashmap = defaultdict
def dfs(x,y,target):
    st[x][y]=True
    g[x][y]=6
    dx = [-1,0,1,0]
    dy = [0,1,0,-1]

    for i in range(4):
        a = x+dx[i]
        b = y+dy[i]
        if(a>=0 and a<5 and b>=0 and b<5 and g[a][b]==target and not st[a][b]):
            dfs(a,b,target)
g = []
st =[[False]*5 for i in range(5)]
for i in range(5):
    g.append(list(map(int,input().split())))
for i in range(5):
    dfs()

