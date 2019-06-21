n = ['a','b','c','d']
n = n+n[0:1]
for x,y in zip(n, n[1:]):
    print (x,y)