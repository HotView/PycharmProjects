# 对于两个子序列，子问题与原问题之间的关系，如果某一个串减少1，对原问题的解有影响吗？
# 转化为二维的网格，横轴为A序列，纵轴为B序列
# 最长的子串就是对于每个单元格的横-纵元素相等时，其左上角的元素值+1就是本身的数值
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
