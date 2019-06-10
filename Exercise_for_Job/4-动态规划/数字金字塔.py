# 问题的关键就是达到顶端只有两种可能：经过次顶层的两个结点
# 如果从底部走到次顶层的最快路径被找到了，只要在这两个选一个就可以了
# 依次类推
def min_weight(tower):
    Maxsum = []
    for row in tower:
        Maxsum.append(row)
    row = len(tower)
    for i in range(1,row):
        for j in range(len(tower[i])):
            Maxsum[i][j] = min(Maxsum[i-1][j],Maxsum[i-1][j+1])+tower[i][j]
    return Maxsum
tower = [[14,45,9,11],[34,18,30],[20,33],[45]]
print(min_weight(tower))


