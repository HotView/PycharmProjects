#逻辑操作，就是在掩码不为零的地方做位操作。
import cv2
import numpy as np
def logic_image(image1,image2):
    dit = cv2.bitwise_and(image1,image2)
    cv2.imshow("and_image",dit)
    print(dit)
def contra_image(image,c,b):
    h,w,ch = image.shape
    blank = np.zeros([h,w,ch],image.dtype)
    cv2.addWeighted(image,c,blank,1-c,b)
image1 = cv2.imread("5.jpg")
image2 = cv2.imread("6.jpg")
logic_image(image1,image2)

cv2.waitKey()