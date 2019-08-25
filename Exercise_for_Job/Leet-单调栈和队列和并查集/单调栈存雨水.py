# 左边第一比它大的数字的位置
heights= list(map(int,input().split()))
res= 0
stk = []
for i in range(len(heights)):
    last = 0
    while(len(stk) and heights[stk[-1]]<=heights[i]):
        #存放的是位置
        t = stk[-1]
        stk.pop()
        res+=(i-t-1)*(heights[t]-last)
        print(heights[t] - last, "#")
        last = heights[t]
    if(len(stk)):
        res+=(i-stk[-1]-1)*(heights[i]-last)
    stk.append(i)
print(res)
