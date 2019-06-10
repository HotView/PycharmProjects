import sys
while True:
    n = sys.stdin.readline().strip()
    if n == ' ':
        break
    lst = sys.stdin.readline().split(' ')
    lst = list(map(int,lst))
    s =  0
    for i in range(len(lst)- 1):
        lst[i+1] = lst[i]+lst[i+1]
        s+= abs(lst[i])
    print(s)