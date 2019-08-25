# 分解+求解+合并
def partition(seq):
    pi,seq = seq[0],seq[1:]
    lo = [x for x in seq if x<=pi]
    hi = [x for x in seq if x> pi]
    return lo,pi,hi
def quicksort(seq):
    if len(seq)<=1:
        return seq
    lo,pi,hi = partition(seq)
    return quicksort(lo)+[pi]+quicksort(hi)
a= [8,5,5,5,5,54,1,2,21,21,13,13,2,0,784]
res = quicksort(a)
print(res)