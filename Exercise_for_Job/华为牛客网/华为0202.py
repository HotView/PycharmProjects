import re
str_data = input()
out_str = ''
def quchu(str_data):
    number = re.findall(r'(\d+)[\(\{\[][A-Za-z]+[\)\}\]]',str_data)
    print(number)
    char_ = re.findall(r'\d+[\[\(\{]([A-Za-z]+)[\}\]\)]',str_data)
    print(char_)
    split_str = re.split(r'\d+[\(\{\[][A-Za-z]+[\)\}\]]',str_data)
    print(split_str)
    out_str = ''
    number = list(map(int,number))
    for i,str in enumerate(char_):
        mul_str = str*number[i]
        out_str =out_str+split_str[i]+mul_str
    out_str = out_str+split_str[-1]
    print(out_str)
    return out_str
out_str = quchu(str_data)
while(True):
    if '[' in out_str or '{'in out_str or '('in out_str:
        out_str = quchu(out_str)
    else:
        break
res_str = ''
leng = len(out_str)
for i in range(leng-1,-1,-1):
    res_str = res_str+out_str[i]
print(res_str)
