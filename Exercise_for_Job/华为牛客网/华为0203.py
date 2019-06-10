import re
str_data = input()
out_str = ''

number = re.findall(r'(\d+)[\(\{\[][A-Za-z]+[\)\}\]]',str_data)
print(number)
char_ = re.findall(r'\d+[\[\(\{]([A-Za-z]+)[\}\]\)]',str_data)
print(char_)
split_str = re.split(r'\d+[\(\{\[][A-Za-z]+[\)\}\]]',str_data)
print(split_str)