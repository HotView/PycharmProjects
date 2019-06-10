import cv2
import numpy as np

vertices = cv2.boxPoints(((200,200),(90,160),60.0))
print(vertices.dtype)
print(vertices)
img = np.zeros((400,400),np.uint8)
for i in range(4):
    p1 = vertices[i,:]
    j = (i+1)%4
    p2 = vertices[j,:]
    cv2.line(img,(p1[0],p1[1]),(p2[0],p2[1]),255,2)
points = np.array([[1,1],[5,1],[1,10],[5,10],[2,5]],np.int32)
# 计算点集的最小外包圆
circle = cv2.minEnclosingCircle(points)
print(circle)
certer1= int(circle[0][0])
certer2 = int(circle[0][1])
R = int(circle[1])
cv2.circle(img,(certer1,certer2),R,126,1)
cv2.boundingRect
cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()