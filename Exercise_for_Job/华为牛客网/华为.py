data = input().split()
str_data = []
def constr(str):
    count = 8-len(i)
    con_str = ''
    while(count):
        con_str =con_str+'0'
        count = count-1
    return i+con_str
for i in data[1:]:
    if len(i)<=8:
        str_data.append(constr(i))
    else:
        while(len(i)>8):
            str_data.append(i[0:8])
            i = i[8:]
        str_data.append(constr(i))
sort_data = sorted(str_data,key=lambda x:x[0])
str_out = ''
for ele in sort_data:
    str_out = str_out+ele+' '
print(str_out)

