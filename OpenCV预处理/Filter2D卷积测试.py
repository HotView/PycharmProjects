import numpy as np
import cv2

test = np.zeros((20,20))
test[10,:] = 1
blur = cv2.blur(test,(1,3))*3#.astype(np.int)
print(test)
print(blur)
for j in range(-2, 3):
    print(j)