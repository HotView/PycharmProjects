import re

def list2str(list0):
    a = ""
    for i in list0:
        a = a + i
        a = a + '\t'
    return a+'\n'
a = open('Tang.txt','r')
flag = True
i = 0
txt_numpy = a.readlines()
#print(txt_numpy)
i = 0
b = open('Data_TangXY.xy','w')
for str0 in txt_numpy:
    if i<2:
        #b.write(str0)
        i = i+1
        continue
    #print(txt_numpy[:3])
    element_file = re.split('[\t\n]',str0)
    flo = float(element_file[1])
    intdata = int(flo)
    element_file[1] = str(intdata)
    data_changed =list2str(element_file)
    print(data_changed)
    b.write(data_changed)
    print(element_file)
    i = i+1
print(i)
b.close()