from matplotlib import pyplot as plt
import numpy as np
import cv2
img = cv2.imread("book01.jpg")
calhist = cv2.calcHist([img],[0],None,[256],[0,256])
plt.plot(calhist,color = 'b')
plt.hist(img.ravel(),400,[0,400])
plt.show()

cv2.waitKey()
cv2.destroyAllWindows()