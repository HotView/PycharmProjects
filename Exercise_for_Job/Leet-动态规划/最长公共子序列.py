# 问题的解会随着单词的长短发生变化
# 所以单词的长度是状态量
# 状态量的两个维度就是单词本身
a="fosh"
b="forsth"
row = len(a)
col = len(b)
f= [[0]*(col+1) for i in range(row+1)]
for i in range(1,row+1):
    for j in range(1,col+1):
        f[i][j] = max(f[i-1][j],f[i][j-1])
        if(a[i-1]==b[j-1]):
            f[i][j] = max(f[i][j],f[i-1][j-1]+1)
print(f[row][col])