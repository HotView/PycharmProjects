n,k = [int(i) for i in input().split('')]
lst = [int(i) for i in input().split('') if i !=0 ]
lst.sort()
n = len(lst)
def solution(lst,k,n):
    index=offset=0
    for i in range(k):
        while index<n and lst[index] == offset:
            index+=1
        if index ==n:
            print(0)
            continue
        print(lst[index]-offset)
        offset = lst[index]