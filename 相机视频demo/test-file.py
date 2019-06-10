import cv2
import matplotlib.pyplot as plt

img = cv2.imread('test.jpg')
plt.figure()
plt.imshow(img)
img_cha = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(img_cha)
plt.show()