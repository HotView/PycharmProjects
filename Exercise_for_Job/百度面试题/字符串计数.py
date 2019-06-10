string=input()
container=[]
container.append(string)
length=len(string)
count=1
tmp_str = string[:]
for i in range(length-1):
    flag = True
    tmp_str = tmp_str[1:]+tmp_str[0]
    #print(tmp_str)
    for j in container:
        if j==tmp_str:
            flag = False
    if flag:
        count +=1
        container.append(tmp_str)
print(count)


