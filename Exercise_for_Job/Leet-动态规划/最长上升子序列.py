a = []
n = len(a)
f = [0]*(n+1)
for i in range(1,n+1):
    f[i] =1
    for j in range(1,i):
        if a[j]<a[i]:
            f[i]  =max(f[i],f[j]+1)
res = max(f)