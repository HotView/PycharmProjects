import cv2
import numpy as np
A = cv2.imread("01.jpg",1)
A[100:500,:] = 255
cv2.imshow("winnm",A)
cv2.waitKey()
cv2.destroyAllWindows()