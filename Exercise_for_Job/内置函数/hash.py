#随机的哈希值，不能用作散列函数
a= 'fejk'
b= hash(a)
print(b)
my_dict= dict()
my_dict[b] = a
print(my_dict)