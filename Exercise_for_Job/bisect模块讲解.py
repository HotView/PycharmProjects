import bisect
a= [32,254,5,151,2,151,2]
a.sort()
print(a)
posi = bisect.bisect(a,2)
pois2 = bisect.bisect_left(a,2)
print(posi,"靠右边的值")
print(pois2,"靠左边的值")
print(a)
bisect.insort(a,87)
print(posi)
print(a)