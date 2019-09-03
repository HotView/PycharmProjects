# 枚举区间长度
s = []
n = len(s)
f = [[0]*(n+1) for i in range(n)]
sum = [0]*(n+1)
for i in range(1,n+1):
    sum[i]+=s[i-1]

for len in range(2,n+1):
    for i in range(1,n+2-len):
        l = i
        r = i+len-1
        f[l][r] = 1e8
        for k in range(l,r):
            f[l][r] = min(f[l][r],f[l][k]+f[k+1][r]+s[r]-s[l-1])
res= f[1][n]