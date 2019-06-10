import cv2
import numpy as np
def fill_color_demo(image):
    copyimg = image.copy()
    h,w = image.shape[:2]
    mask = np.zeros([h+2,w+2],np.uint8)
    cv2.FLOODFILL_FIXED_RANGE
    cv2.FLOODFILL_MASK_ONLY