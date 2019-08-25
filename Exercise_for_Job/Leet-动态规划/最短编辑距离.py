a= []
b=[]
n = len(a)
m= len(b)
f= [[]]
for  i in range(m+1):
    f[0][i] = i
for i in range(m + 1):
    f[i][0] = i
for i in range(n+1):
    for j in range(m+1):
        f[i][j] = min(f[i-1][j],f[i][j-1])
        if(a[i]==b[j]):
            f[i][j] = min(f[i][j],f[i-1][j-1])
        else:
            f[i][j] = min(f[i][j],f[i-1][j-1]+1)
