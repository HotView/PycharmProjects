import numpy as np
import cv2


from OpenCV相机标定.camera import raw_camera_matrix
from OpenCV相机标定.camera import distortion_coefficients
from OpenCV相机标定.camera import undistort_camera_matrix
from OpenCV相机标定.camera import image_size
from OpenCV相机标定.camera import roi

img = cv2.imread("image/left12.jpg")
dst = cv2.undistort(img,raw_camera_matrix,distortion_coefficients,None,undistort_camera_matrix)
cv2.imshow("origin",img)
cv2.imshow("res",dst)
cv2.waitKey(0)