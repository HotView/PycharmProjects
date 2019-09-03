# 集合：f[u][0]所有以u为根节点的子树的方案,且不选这个点，儿子可以选也可以不选
# 集合：f[u][1]所有以u为根节点的子树的方案,且选这个点，儿子不能选
# 属性：Max
#两种状态
from collections import defaultdict
import sys
sys.setrecursionlimit(100000)
def dfs(u):
    dp[u][1] = happy[u]
    sons = hashson[u]
    for i in range(sons):
        #print(one,"-->",u)
        one = sons[i]
        dfs(one)
        dp[u][0] +=max(dp[one][0],dp[one][1])
        dp[u][1]+=dp[one][0]
def find(x):
    if(p[x]!=x):
        p[x] =find(p[x])
    return p[x]
N = 6010
happy = [0]*N
hashson = defaultdict(list)
st = [False]*N
n = int(input())
if n==1:
    print(1)
    sys.exit()
p = [i for i in range(n+1)]
for i in range(n):
     happy[i+1] = int(input())
for i in range(n-1):
    s,f = list(map(int,input().split()))
    p[s] =f
    hashson[f].add(s)
dp= [[0]*2 for i in range(n+1)]
root= find(n)
dfs(root)
print(max(dp[root][0],dp[root][1]))