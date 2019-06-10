import cv2
import numpy as np
from numpy import linalg as la
img = cv2.imread('01.jpg',0)
print(img.shape)
u,s,vt = la.svd(img)
#u0,s0,vt0 = la.svd(img[:,:,0])
#u1,s1,vt1 = la.svd(img[:,:,1])
#u2,s2,vt2 = la.svd(img[:,:,2])
#sigma_T = np.zeros((len(s),len(s)))
def get_approx_matrix(u,s,vt,rank):
    m = len(u)
    n = len(vt)
    uv = np.zeros((m,n))
    for i in range(rank):
        U = u[:,i]
        U = U.reshape(m,1)
        V = vt[i]
        V = V.reshape(1,n)
        uv+= s[i]*np.dot(U,V)
    uv[uv<0] = 0
    uv[uv>255] = 255
    print(sum(uv[uv<0]))
    return uv.astype("uint8")
number = 50
uv = get_approx_matrix(u,s,vt,number)
#uv1 = get_approx_matrix(u1,s1,vt1,number)
#uv2 = get_approx_matrix(u2,s2,vt2,number)
#uv = np.stack((uv0,uv1,uv2),2)
cv2.imshow("origin1",img)
cv2.imshow("recover",uv)
cv2.waitKey()
cv2.destroyAllWindows()

