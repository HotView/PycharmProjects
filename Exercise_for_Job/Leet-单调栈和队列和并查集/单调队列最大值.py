nums = list(map(int,input().split()))
k = int(input())
q = []
res  = []
n = len(nums)
for i in range(n):
    if(len(q) and i-k-1>q[0]):
        #出窗口删除
        q.pop(0)
    while(len(q) and nums[q[-1]]<= nums[i]):
        #递减的序列，如果成立就正相关，单调递增。
        q.pop()
    q.append(i)
    if (i>=k-1):
        #最大值在队头，单调递减的队列
        res.append(nums[q[0]])
print(res)