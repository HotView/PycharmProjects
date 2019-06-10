import matplotlib.pyplot as plt
import numpy as np
import cv2
img = cv2.imread('1.jpg')
print(type(img))
test_img = img[:,:,0]

test_arry = test_img.ravel()
hahha = np.append(test_arry,[1000]*400)
#plt.figure(1)
# print(hahha)
# plt.hist(hahha,10,[0,100])
# plt.figure(2)
# plt.hist(test_img.ravel(),256,[0,256],)
# #plt.show()
plt.figure(2)
hist = cv2.calcHist(img,[1],None,[256],[0,256])
plt.plot(hist)
plt.show()

