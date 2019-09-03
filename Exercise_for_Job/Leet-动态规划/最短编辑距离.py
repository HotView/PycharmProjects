#三种子状态
# a中删除一个：f[i-1][j]+1
# a中增加一个:f[i][j-1]+1
# a中修改一个:f[i-1][j-1]+1/0
a= []
b=[]
n = len(a)
m= len(b)
f= [[]]
#初始化更新操作
for  i in range(m+1):
    f[0][i] = i
for i in range(m + 1):
    f[i][0] = i
for i in range(1,n+1):
    for j in range(1,m+1):
        f[i][j] = min(f[i-1][j]+1,f[i][j-1]+1)
        if(a[i]==b[j]):
            f[i][j] = min(f[i][j],f[i-1][j-1])
        else:
            f[i][j] = min(f[i][j],f[i-1][j-1]+1)
print(f[n][m])