import cv2
import numpy as np
from scipy import ndimage

kernel_3x3 = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
kernel_5x5 = np.array([[-1,-1,-1,-1,-1],[-1,1,2,1,-1],[-1,2,4,2,-1],[-1,1,2,1,-1],[-1,-1,-1,-1,-1]])
image = cv2.imread('3.jpg',0)


k3 = ndimage.convolve(image,kernel_3x3)
k5 = ndimage.convolve(image,kernel_5x5)
k3_ = cv2.filter2D(image,-1,kernel_3x3)
cv2.imshow("origin",image)
cv2.imshow("33-convolve",k3)
cv2.imshow("33-filter2D",k3_)
cv2.waitKey()
cv2.destroyAllWindows()