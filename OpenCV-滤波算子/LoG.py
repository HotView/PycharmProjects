import cv2
import numpy as np
def creatLoGkernel(sigma,size):
    H,W = size
    r,c = np.mgrid[0:H:1,0:W:1]
    r=r-(H-1)/2
    c=c-(W-1)/2
    sigma2 = pow(sigma,2.0)
    norm2 = np.power(r,2.0)+np.power(c,2.0)
    LoGKernel = (norm2/sigma2-2)*np.exp(-norm2/(2*sigma2))
    return LoGKernel

print(creatLoGkernel(0.5,(3,3)))
for i in range(1,10):
    j = i/10
    print(creatLoGkernel(j, (5, 5)))
