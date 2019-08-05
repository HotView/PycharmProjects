import re
a="d875412e2362"
b = "2321gfr52"
c = "e2321gfr52"
d = "ede2321gfr52"
print(re.findall('^d?\d+',a))
print(re.findall('^d?\d+',b))
print(re.findall('^[ed]?\d+',c))
print(re.findall('^[ed]?\d+',d))
print(re.findall('\d+',a),"fhd")

a = 100
b = ~a+1
print(a,b)
print(range(-4,4))
finall = re.findall('^[ed]?(\d+)',c)
print(finall)
help(dict.pop)