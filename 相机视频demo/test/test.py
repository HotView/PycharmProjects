import matplotlib.pyplot as plt
import cv2
img = cv2.imread('test.jpg',0)
_,ax = plt.subplots(3,4)
ax4 = ax[1][2]
ax4.imshow(img)
print(dir(ax4))
#help(ax4)
plt.figure(2)
plt.show()