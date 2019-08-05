# 有 N 种物品和一个容量为 W 的背包，每种物品都有无限件可用
# 重量为W，价值为V
# 状态变量为F[i][j]，i为物品，j为背包容量
# 背包的核心问题：F[j] = max(F[j],F[j-W[i]+V[i]])，即不选与选。
class CompletePack():
    def __init__(self,v =[1500, 2000, 3000] ,w_cost= [1, 3, 4] ,target = 4):
        self.v = v # Value
        self.w_cost = w_cost  # Weight
        self.len = len(V)
        self.target = target
        self.F = [0 for i in range(target+1)]
    def Complete(self,cost, value):
        for i in range(cost,self.target+1):
            self.F[i] = max(self.F[i],self.F[i-cost]+value)
    def solution(self):
        for i in range(self.len):
            self.Complete(self.w_cost[i],self.v[i])
        return self.F[self.target]
W = [3,2,6,7,1,4,9,5]
V = [6,3,5,8,3,1,6,9]
target = 15
F = [0 for i in range(0,target+1)]
n = len(W)
def CompleteBackPack(cost,value):
    for i in range(cost,target+1):#这是和01背包唯一的区别：正序遍历！
        F[i] = max(F[i],F[i-cost]+value)
for i in range(n):
    CompleteBackPack(W[i],V[i])
print(max(F))
pack = CompletePack(V,W,target)
print(pack.solution())
