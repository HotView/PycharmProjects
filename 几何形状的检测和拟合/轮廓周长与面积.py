import cv2
import numpy as np

points = np.array([[[0,0]],[[50,30]],[[100,0]],[[100,100]]],np.float32)
length1 = cv2.arcLength(points,False)
length2 = cv2.arcLength(points,True)
area = cv2.contourArea(points)
print(length1,length2)
print("area",area)
img = np.zeros((200,200),dtype=np.uint8)
rect = cv2.boundingRect(points)
print(rect)
print(points[0])
for i in range(len(points)):
    cv2.line(img,tuple(points[i][0]),tuple(points[(i+1)%4][0]),255)
cv2.rectangle(img,(rect[0],rect[1]),(rect[2],rect[3]),128)
cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
