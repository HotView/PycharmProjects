from matplotlib import pyplot as plt
import numpy as np
import cv2
img = cv2.imread("book01-.jpg")
img_cut = img[1000:,:]
img_resize = cv2.resize(img_cut,(600,600),interpolation=cv2.INTER_CUBIC)
#cv2.imwrite("book01resize-.jpg",img_resize)
cv2.imshow("origin",img)
cv2.imshow("reszie",img_resize)
cv2.waitKey()
cv2.destroyAllWindows()