# N件物品，和一个容积为M的背包
# 第i件物品的体积W[i]，价值是v[i]
# 求解装入哪些物品是的价值总和最大
# 对于第i件物品的态度是拿还是不拿？
# 背包的核心问题：F[j] = max(F[j],F[j-W[i]]+V[i]])
from itertools import chain
def oneSolution():
    V = [1500,2000,3000]    #Value
    W = [1,3,4] #Weight
    N = 4
    row = len(W)
    F = [[0 for i in range(N+1)] for j in range(row)]
    for i in range(row):
        for j in range(W[i],N+1):
            F[i][j] = F[i-1][j]
            F[i][j] = max(F[i][j],F[i-1][j-W[i]]+V[i])
            print(F[i][j])
    res= 0
    for i in list(chain.from_iterable(F)):
        res = max(res,i)
    print(res)

class OneZeroPack():
    def __init__(self,v =[1500, 2000, 3000] ,w_cost= [1, 3, 4] ,target = 4):
        self.v = v # Value
        self.w_cost = w_cost  # Weight
        self.len = len(V)
        self.target = target
        self.F = [0 for i in range(target+1)]
    def Complete(self,cost, value):
        for i in reversed(range(cost,self.target+1)):
            self.F[i] = max(self.F[i],self.F[i-cost]+value)
    def solution(self):
        for i in range(self.len):
            self.Complete(self.w_cost[i],self.v[i])
        return self.F[self.target]
target = 1000
V = [6,10,3,4,5,8]
W = [200,600,100,180,300,450]
pack = OneZeroPack(v= V,w_cost = W,target=target)
print(pack.solution())
#oneSolution()
