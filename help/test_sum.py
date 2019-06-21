import cv2
import numpy as np

a= [1,2,3,4,5,7]
b =[1,2,8,4,8,6]
c = sum(x == y for x,y in zip(a,b))
print(c)

d= True
f = True
g = d==f
print(g)