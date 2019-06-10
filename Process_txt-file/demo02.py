file = open("02.txt")
a = file.readline()
print(a)
while a:
    a = a.split()
    print("\""+a[0]+'\"'+',')
    a = file.readline()