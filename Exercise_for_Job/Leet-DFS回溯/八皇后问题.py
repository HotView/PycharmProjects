# 从第一行开始看
# 枚举每一行，全排列问题
# 92种可能方案

N= 20
col = [False]*N
dg = [False]*N
udg = [False]*N
st = [False]*N
n = 4
g = [["."]*n for i in range(n)]
cnt = 0
def dfs(u):
    global cnt
    if(u==n):
        cnt+=1
        for i in range(n):
            for j in range(n):
                print(g[i][j], end="")
            print()
        print("##############")
    for i in range(n):
        if(not col[i] and not dg[u+i] and not udg[i-u+n]):
            g[u][i] = 'Q'
            col[i] = dg[u+i] = udg[i-u+n] =True
            dfs(u+1)
            col[i] = dg[u+i] = udg[i-u+n] = False
            g[u][i] ="."
dfs(0)
print(cnt)

