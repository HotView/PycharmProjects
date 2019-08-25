# 背包多重问题：每种物品都有数量为ni
# 可以将问题分解为0-1背包问题和完全背包问题的组合解！
# 背包的核心问题：F[j] = max(F[j],F[j-W[i]+V[i]])，即不选与选。
class MultiplePack():
    def __init__(self,Wlist = [3,2,6,7,1,4,9,5],Vlist =[6,3,5,8,3,1,6,9],Count =  [3,5,1,9,3,5,6,8],target = 20 ):
        self.V = Vlist
        self.W = Wlist
        self.target = target
        self.Count = Count
        self.len = len(Wlist)
        self.F = [0 for i in range(target+1)]
    def OneZeroPack(self,cost,value):
        for i in reversed(range(cost,self.target+1)):
            self.F[i] = max(self.F[i],self.F[i-cost]+value)
    def CompletePack(self,cost,value):
        for i in range(cost,self.target+1):
            self.F[i] = max(self.F[i],self.F[i-cost]+value)
    def MultiplePack(self,cost,value,count):
        if (cost*count)>self.target:
            self.CompletePack(cost,value)
            return
        tmpcount = 1
        while(tmpcount<count):
            self.OneZeroPack(cost*tmpcount,value*tmpcount)
            #count = count-tmpcount #下面有优化的示例，即运用对数来计算即可
            tmpcount= tmpcount+1
        if cost*count==self.target:
            self.OneZeroPack(cost*count,value*count)
    def Solution(self):
        for i in range(self.len):
            self.MultiplePack(self.W[i],self.V[i],self.Count[i])
        return self.F
pack = MultiplePack()
print(pack.Solution())
#
#
#
C = [3,2,6,7,1,4,9,5]
V = [6,3,5,8,3,1,6,9]
Count = [3,5,1,9,3,5,6,8]#每种物品的实现
target = 20
F = [0 for i in range(0,target+1)]
n = len(C)
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

for i in range(0,n):
    MultipleBackPack(C[i],V[i],Count[i])
print (F)



