#选出区间，求取数值*sum（区间）
import sys
n = sys.stdin.readline()
lst = [int(i) for i in sys.stdin.readline().split()]
index = 0
res= 0
leng = len(lst)
left = right = 0
for num in lst:
    left = right = index
    num_sun = 0
    #next_left_num = lst[left-1]
    while(left-1>=0) and lst[left-1]>=num:
        left = left-1
        num_sun = num_sun+lst[left-1]
    #next_right_num  = lst[right+1]
    for num_r in lst[right+1:]:
        if num_r<num:
            break
        right = right+1
        num_sun = num_sun+num_r
    #right = right-1
    print(index,left,right)
    index  =index+1
    print(num,"*",lst[left:right+1])
    res = max(res,num*(num+num_sun))
print(res)
