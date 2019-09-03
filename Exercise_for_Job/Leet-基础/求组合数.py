import math

def factorial_(n):
    result=1
    for i in range(2,n+1):
        result=result*i
    return result

def comb_1(n,m):
    return math.factorial(n)//(math.factorial(n-m)*math.factorial(m))  #直接使用math里的阶乘函数计算组合数

def comb_2(n,m):
    return factorial_(n)//(factorial_(n-m)*factorial_(m))              #使用自己的阶乘函数计算组合数

def perm_1(n,m):
    return math.factorial(n)//math.factorial(n-m)                        #直接使用math里的阶乘函数计算排列数

def perm_2(n,m):
    return math.factorial(n)//math.factorial(n-m)                        #使用自己的阶乘函数计算排列数

if __name__=='__main__':
    print(comb_1(5,2))
    print(comb_2(3,2))
    print(perm_1(3,2))
    print(perm_2(3,2))
