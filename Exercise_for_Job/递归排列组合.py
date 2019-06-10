# m个苹果放n个盘子，盘子苹果无差别！
# 允许盘子为空的答案
# 不允许为空的答案，可以先预先在每个盘子里放置一个苹果，把问题转化为盘子可空的情况，苹果数减去盘子数。
def AllowEmpty(m,n):
    if n>m:
        return AllowEmpty(m,m)
    if m==0:
        return 1
    if n==0:
        return 0
    return AllowEmpty(m,n-1)+AllowEmpty(m-n,n)
def NoEmpty(m,n):
    solution = AllowEmpty(m-n,n)
    return solution
print(NoEmpty(5,5))
print(AllowEmpty(7,3))