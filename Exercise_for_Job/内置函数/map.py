#Return an iterator that applies function to every item of iterable, yielding the results.
#map对象也是迭代器，可以用list换为列表,或用元组换为元组，或''.join（map）换为字符串
#Make an iterator that computes the function using arguments from each of the iterables.
a = '4565464454'
b =list(a)
c = map(int,b)
help(c)
print(c)
tuple_ = tuple(b)
print(tuple_)
print(list(c))
str_  =''.join(c)
print(str_)