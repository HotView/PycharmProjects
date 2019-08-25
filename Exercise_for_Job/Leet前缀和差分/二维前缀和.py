n,m = list(map(int ,input().split()))
a = input()
s= [[0]*m for i in range(n)]
#求解前缀和
for i in range(1,n+1):
    for j in range(1,m+1):
        s[i][j] = s[i-1][j]+s[i][j-1]-s[i-1][j-1]+a[i][j]


print()