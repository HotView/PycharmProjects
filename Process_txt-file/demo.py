file  = open('01.txt')
a = file.readline()
appid =""
while a:
    c = a.split()
    appid = appid + c[-1]+'|'
    b ="    "+"\""+c[-1]+"\""+','
    print(b)
    a = file.readline()

print('#####33')
print(appid)