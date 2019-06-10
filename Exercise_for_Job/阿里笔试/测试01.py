import re

sent = input()
str1 = re.findall(r'<.*?>',sent)
print(str1)
str2 = re.findall(r'\[.*?\]',sent)
print(str2)
#print(sent[0:42])