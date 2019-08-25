# 问题的解会随着单词的长短发生变化
# 所以单词的长度是状态量
# 状态量的两个维度就是单词本身
A="asdhfkjhgfdsakolfr"
B="djkhfkjhgfdsaqwe"
row = len(A)
col = len(B)
cell = [[0 for i in range(col)] for j in range(row)]
for i,char in enumerate(A):
    if char == B[0]:
        cell[0][i] = 1
for i,char in enumerate(B):
    if char==A[0]:
        cell[i][0] = 1
for i in range(1,row):
    for j in range(col):
        if A[i]==B[j]:
            cell[i][j] = cell[i-1][j-1]+1
        else:
            cell[i][j] = 0
print(cell)