# 给n个字符串
# 给定一个模式
# 询问n个和匹配模式的距离
N = 20
f = [[0]*N for i in range(N)]
def edit_dis(s1,s2):
    l1 = len(s1)
    l2 = len(s2)
    for i in range(l2+1):
        f[0][i] = i
    for i in range(l1 + 1):
        f[i][0] = i
    for i in range(1,l1):
        for j in range(1,l2):
            f[i][j] = min(f[i-1,j-1]+1,f[i][j-1]+1)
            if l1[1]==l2[j]:
                f[i][j] = min(f[i][j],f[i-1][j-1])
            else:
                f[i][j] = min(f[i][j],f[i-1][j-1]+1)