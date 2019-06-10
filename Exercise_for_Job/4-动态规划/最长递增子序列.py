# 分析问题：一个长度为 N 的序列去掉最后一个元素之后，与原问题解之间的关系？
# 可以遍历前 N-1 的可能的序列，然后如果其最后一个元素大于第 N 个数字，则加+，然后对所有的 N-1 可能子序列取最大值
def basic_lis(seq):
    length = len(seq)
    L = [1 for i in range(length)]
    for i in range(1,length):
        for j in range(i):
            if seq[i]>seq[j]:
                L[i] = max(L[j]+1,L[i])
    return L
a = [1,5,56,6,5,56,5,56,78]
len = basic_lis(a)
print(len)