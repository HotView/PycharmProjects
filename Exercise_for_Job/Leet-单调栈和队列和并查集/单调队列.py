# 从队头取最值，队尾插入元素
# 判断队头是否划出窗口。
# 判断新插入元素之后是否满足单调性。
# 取队头元素。
nums = list(map(int,input().split()))
k = int(input())
q = []
res  = []
n = len(nums)
for i in range(n):
    ## 判断是否划出窗口
    if(len(q) and i-k-1>q[0]):
        q.pop(0)
    ##判断是否满足单调性，单调递减
    while(len(q) and nums[q[-1]]<= nums[i]):
        q.pop()
    q.append(i)
    if (i>=k-1):
        #最大值在队头，单调递减的队列
        res.append(nums[q[0]])
print(res)