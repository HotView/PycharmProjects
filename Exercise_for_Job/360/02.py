n,m = map(int,input().split())
num1 = list(map(int,input().split()))
num2 = list(map(int,input().split()))
res = []
dict()
for i in range(n):
    num_max = 0
    temp1 = 0
    temp2 = 0
    for x in num1:
        for y in num2:
            if ((x+y)%m)>num_max:
                num_max =(x+y)%m
                temp1 = x
                temp2 = y
    num1.remove(temp1)
    num2.remove(temp2)
    print(num_max, end=' ')



