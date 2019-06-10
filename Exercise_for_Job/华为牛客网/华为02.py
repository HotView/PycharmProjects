import re
#print('a'*3)
str_data = input()
number = re.findall(r'\d+',str_data)
char_ = re.findall(r'\(([A-z0-9\(\)\{\}\[\]]+)\)?',str_data)
split_str = re.split(r'\d+\([A-z0-9\(\)\{\}\[\]]+\)',str_data)
#print(split_str)
number = list(map(int,number))
out_str = ''
for i,str in enumerate(char_):
    mul_str = str*number[i]
    out_str =out_str+split_str[i]+mul_str
leng = len(out_str)
#print(leng)
res_str = ''
for i in range(leng-1,-1,-1):
    res_str = res_str+out_str[i]
print(res_str)
#for ele_str in char_:


#index_number = re.findall(r'\d\(',str_data)