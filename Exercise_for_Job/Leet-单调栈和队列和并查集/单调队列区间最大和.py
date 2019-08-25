data = list(map(int,input().split()))
n = len(data)
data =data+data
res = -100000
presum= [0]*(2*n+1)
for i in range(1,2*n+1):
    presum[i] = presum[i-1]+data[i-1]
q = []
for i in range(1,2*n+1):
    if(len(q) and i-n>q[0]):
        q.pop(0)
    # 最值存储在队头位置
    if(len(q)):
        res = max(res,presum[i]-presum[q[0]])
    if(len(q) and presum[q[-1]]>=presum[i]):
        q.pop()
    q.append(i)
print(res)
