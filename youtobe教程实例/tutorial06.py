#+-*/运算
import cv2
import numpy as np
def add_image(image1,image2):
    dit = cv2.add(image1,image2)
    cv2.imshow("add_inage",dit)

def sub_image(image1,image2):
    dit = cv2.subtract(image1,image2)
    cv2.imshow("sub_image",dit)
def mul_image(image1,image2):
    dit = cv2.multiply(image1,image2)
    cv2.imshow("mul_image",dit)
def div_image(image1,image2):
    dit = cv2.divide(image1,image2)
    cv2.imshow("div_image",dit)
image1 = cv2.imread("5.jpg")
image2 = cv2.imread("6.jpg")
add_image(image1,image2)
sub_image(image1,image2)
mul_image(image1,image2)
div_image(image1,image2)
cv2.waitKey()