# 从栈顶取最值
def solution():
    heights = list(map(int,input().split()))
    n = len(heights)
    left = [0 for i in range(n)]
    right= [0 for i in range(n)]
    stk = []
    for i in range(n):
        # 单调递增栈
        while(stk and heights[stk[-1]]>=heights[i]):
            stk.pop()
        if(not stk):
            left[i]  =-1
        else:
            left[i] = stk[-1]
        stk.append(i)
    while stk:
        stk.pop()
    print(stk)
    for i in range(n-1,-1,-1):
        while(stk and heights[stk[-1]]>=heights[i]):
            stk.pop()
        if(not stk):
            right[i] = n
        else:
            right[i] = stk[-1]
        stk.append(i)
    res = 0
    for i in range(n):
        res = max(res,heights[i]*(right[i]-left[i]-1))
    return res