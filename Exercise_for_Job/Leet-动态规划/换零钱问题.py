coins = []
target  =100
f =[[0]*10 for i in range(10)]
n = len(coins)
# 先枚举每一种硬币
for i in n:
    for j in range(coins[i],target):
        f[i][j] = f[i-1][j]+f[i][j-coins[i]]
print(f[n-1][target])