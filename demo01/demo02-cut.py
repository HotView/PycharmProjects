import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('book01.jpg',0)
img01 = img[1000:3000,:]
print(img.shape)
print(dir(img))
print(img.argmax)
print(img.ndim)
#print(img.dim)
cv2.imshow("origin",img)
cv2.imshow("cut",img01)
cv2.waitKey()
cv2.destroyAllWindows()