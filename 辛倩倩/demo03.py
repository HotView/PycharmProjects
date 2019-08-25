import heapq
from collections import defaultdict

def get_one_num(num):
    res = 0
    while(num):
        if (num&1):
            res = res+1
        num=num>>1
    return res
print(get_one_num(4))