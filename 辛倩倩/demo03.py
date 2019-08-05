import bisect
#import
a = [6,5,6,5,1,5,5,4,12,56,5,5,61,23]
print(a.index(5))
index  = bisect.bisect(a,89)
#print(a[index-1])
print(len(a))
print(index)
bisect.insort(a,89)
print(a)