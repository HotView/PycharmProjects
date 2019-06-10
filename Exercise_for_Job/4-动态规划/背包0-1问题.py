# N件物品，和一个容积为M的背包
# 第i件物品的体积W[i]，价值是v[i]
# 求解装入哪些物品是的价值总和最大
# 对于第i件物品的态度是拿还是不拿？
# 背包的核心问题：F[j] = max(F[j],F[j-W[i]+V[i]])
def oneSolution():
    V = [1500,2000,3000]    #Value
    W = [1,3,4] #Weight
    N = 4
    row = len(W)+1
    col = N+1
    F = [[0 for i in range(col)] for j in range(row)]
    print(F)
    for i in range(1,row):
        for j in range(1,col):
            F[i][j] = F[i-1][j]
            print("j:",j)
            if j>=W[i-1]:
                F[i][j] = max(F[i-1][j],F[i-1][j-W[i-1]]+V[i-1])
            print(F[i][j])
    print(F)
class OneZeroPack():
    def __init__(self,V =[1500, 2000, 3000] ,W= [1, 3, 4] ,target = 4):
        self.V = V # Value
        self.W = W  # Weight
        self.len = len(V)
        self.target = target
        self.F = [0 for i in range(target+1)]
    def Complte(self,cost, value):
        for i in reversed(range(cost,self.target+1)):
            self.F[i] = max(self.F[i],self.F[i-cost]+value)
    def solution(self):
        for i in range(self.len):
            self.Complte(self.W[i],self.V[i])
        return self.F
target = 1000
V = [6,10,3,4,5,8]
W = [200,600,100,180,300,450]
pack = OneZeroPack(V= V,W = W,target=target)
print(pack.solution())
