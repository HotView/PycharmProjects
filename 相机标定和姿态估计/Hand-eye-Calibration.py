import cv2
import numpy as np
img = np.random.randint(0,255,(30,30))
print(img)
print(cv2.inRange(img,40,100))
print(cv2.cartToPolar(np.array([5.0,4]),np.array([5.0,4])))
def TSAI(Hmaker2world,Hgrid2cam):
    A = []
    n = len(Hgrid2cam)
    Hgij = np.zeros(Hmaker2world.shape)
    Hcij = np.zeros(Hgrid2cam.shape)
    for i in range(n-1):
        Hgij[:,:,i] = cv2.invert(Hmaker2world[:,:,i+1])*Hmaker2world[:,:,i]
        Hcij[:,:,i] = Hgrid2cam[:,:,i+1]*cv2.invert(Hgrid2cam[:,:,i])
        rgij = cv2.Rodrigues()
