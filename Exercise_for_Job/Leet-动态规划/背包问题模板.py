# 01背包问题：从小到大递推
# 完全背包问题：从大到小递推
# 多重背包问题：
def CompleteBackPack(cost,value):
    for i in range(cost,target+1):
        F[i] = max(F[i],F[i-cost]+value)
def OneZeroBackPack(cost,value):
    for i in reversed(range(cost,target+1)):
        F[i] = max(F[i],F[i-cost]+value)
def MultipleBackPack(cost,value,count):
        if (cost * count) >= target:#当该种物品的个数乘以体积大于背包容量，视为有无限个即完全背包
            CompleteBackPack(C[i],V[i])
            return
        temp_count = 1  #以上情况不满足，转化为以下情况，具体参考《背包九讲》多重背包的时间优化
        while(temp_count<count):
            OneZeroBackPack(temp_count*cost,temp_count*value)
            count = count - temp_count
            temp_count = temp_count * 2  #转化为1，2，4，8，这样的系数可以配凑出任意数目的这件物品，优化算法
        OneZeroBackPack(count*cost,count*value)#9个中剩下两个
C = [3,2,6,7,1,4,9,5]
V = [6,3,5,8,3,1,6,9]
Count = [3,5,1,9,3,5,6,8]#每种物品的实现
target = 20
F = [0]*(target+1)
n = len(C)
for i in range(0,n):
    MultipleBackPack(C[i],V[i],Count[i])
print(F)