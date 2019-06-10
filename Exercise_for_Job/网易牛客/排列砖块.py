string = input()
if len(set(string)) == 2:
    print(2)
elif len(set(string))==1:
    print(1)
else:
    print(0)
    