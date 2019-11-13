import numpy as np
from math import pi

def hx(zita,m):
    h = np.zeros((2,m),dtype=np.float)
    for i in range(-m,m):
        for j in range(-m,m):
            zita2 = zita*zita
            h[i+m][j+m] = -(j)/zita2*1/(2*pi*zita2)*np.exp(-((i)^2+(j)^2)/(2*zita2))
    return h
a = hx(4,5)
print(a)
def hy(zita,m):
    pass
def hxx(zita,m):
    pass
def hxy(zita,m):
    pass
def hyy(zita,m):
    pass

