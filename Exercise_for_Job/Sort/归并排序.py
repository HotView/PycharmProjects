# 分解+求解+合并
def mergesort01(seq):
    mid = int(len(seq)/2)
    left,right = seq[:mid],seq[mid:]
    if len(left)>1:left = mergesort01(left)
    if len(right)>1:right = mergesort01(right)
    res = []
    while left and right:
        if left[-1]>=right[-1]:
            res.append(left.pop())
        else:
            res.append(right.pop())
    res.reverse()
    return (left or right)+res
def mergesort(seq):
    if len(seq)<2:
        return seq
    mid = int(len(seq)/2)
    left = mergesort(seq[:mid])
    right = mergesort(seq[mid:])
    return merge(left,right)
def merge(left,right):
    res = []
    while left and right:
        if left[0]<=right[0]:
            res.append(left.pop(0))
        else:
            res.append(right.pop(0))
    res = res+left
    res = res+right
    return res
seq = [5,3,0,6,1,4]
a = mergesort(seq)
print(a)
