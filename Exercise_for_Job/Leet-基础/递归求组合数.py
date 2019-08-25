#c[a][b] 表示从a中选b个的方案数
c= [[]]
n = int(input())
for i in range(n):
    for j in range(i+1):
        if(not j): c[i][j] = 1
        else:
            c[i][j] = c[i-1][j]+c[i-1][j-1]