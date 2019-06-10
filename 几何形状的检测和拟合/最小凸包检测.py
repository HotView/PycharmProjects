import cv2
import numpy as np
s = 400
I = np.zeros((s,s),np.uint8)

n=  80
points=  np.random.randint(100,300,(n,2),np.int32)
for i in range(n):
    cv2.circle(I,(points[i,0],points[i,1]),2,255,2)
convexhull = cv2.convexHull(points)
#print(points[0][1])
#print(points[0,1])
print(type(convexhull))
print(convexhull.shape)
k = convexhull.shape[0]
for i in range(k-1):
    cv2.line(I,(convexhull[i,0,0],convexhull[i,0,1]),(convexhull[i+1,0,0],convexhull[i+1,0,1]),255,1,cv2.LINE_AA)
cv2.line(I,(convexhull[k-1,0,0],convexhull[k-1,0,1]),(convexhull[0,0,0],convexhull[0,0,1]),255,1)
cv2.HoughLines()
cv2.namedWindow("I",0)
cv2.imshow("I",I)
cv2.waitKey(0)
cv2.destroyAllWindows()