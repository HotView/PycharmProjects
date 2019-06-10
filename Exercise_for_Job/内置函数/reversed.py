#Return a reverse iterator. seq must be an object which has a __reversed__() method or supports
#the sequence protocol (the __len__() method and the __getitem__() method with integer arguments
#starting at 0).
#Return a reverse iterator
b ='Zahfkjsdhf'
c= reversed(b)
help(c)
print(''.join(c))