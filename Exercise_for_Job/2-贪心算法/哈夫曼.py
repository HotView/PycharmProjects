from heapq import heapify,heappop,heappush
from itertools import count
import itertools
print(list(itertools.repeat(4,10)))
def hunff(seq,frq):
    num = count()
    trees = list(zip(frq,num,seq))
    print(trees)
    heapify(trees)
    print(trees)
    while len(trees)>1:
        fa,_,a = heappop(trees)
        fb,_,b = heappop(trees)
        n = next(num)
        print(n, trees)
        heappush(trees,(fa+fb,n,[a,b]))
        print(n, trees)
    return trees
seq = "abcdefghi"
frq = [6,5,4,9,20,12,15,16,11]
res =hunff(seq,frq)
print(res)