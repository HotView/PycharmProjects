# 金币面额为：33,24,12,5,1
# 凑出的总数为n
# 输出最少硬币数目
def f_count(n):
    if n<=0:
        return 0
    f = [0]*(n+1)
    for k in range(1,n+1):
        if k<5:
            f[k] = 1+f[k-1]
        elif k<12:
            f[k] = 1+min(f[k-1],f[k-5])
        elif  k<24:
            f[k] = 1 + min(f[k-12],f[k - 1], f[k - 5])
        elif k<33:
            f[k] = 1 + min(f[k - 1], f[k - 5],f[k-12],f[k-24])
        else:
            f[k] = 1+min(f[k - 1], f[k - 5],f[k-12],f[k-24],f[k-33])
    return f[n]
print(f_count(4))
