from collections import defaultdict
import sys
def dfs(u):
    dp[u][1] = happy[u]
    for one in hashson[u]:
        dfs(one)
        dp[u][0] +=max(dp[one][0],dp[one][1])
        dp[u][1]+=dp[one][0]
N = 6010
happy = [1]*N
hashson = defaultdict(list)
n = int(input())
if n==1:
    print(1)
    sys.exit()
father = set()
son = set()
for i in range(n):
     happy[i+1] = int(input())
for i in range(n):
    s,f = list(map(int,input().split()))
    father.add(f)
    father.add(s)
    son.add(s)
    hashson[f].append(s)
dp= [[0]*2 for i in range(N)]
root= father-son
#print(root)
#print(len(root))
root = root.pop()
#print(root)
dfs(root)
print(max(dp[root][0],dp[root][1]))