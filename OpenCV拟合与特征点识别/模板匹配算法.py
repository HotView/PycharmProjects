import cv2
import numpy as np
kernel_left = np.array([[1,1,1,1,1,0,0,0],
                        [1,1,1,1,1,1,0,0],
                        [0,0,0,0,0,1,1,0],
                        [0,0,0,0,0,0,1,1]])
kernel_right = np.array([[0,0,0,1,1,1,1,1],
                         [0,0,1,1,1,1,1,1],
                         [0,1,1,0,0,0,0,0],
                         [1,1,0,0,0,0,0,0]])
img = cv2.imread("center-v1.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype("float")
start = cv2.getTickCount()
res1 = cv2.filter2D(gray,-1,kernel_left)
res2 = cv2.filter2D(gray,-1,kernel_right)
points=  []
max1 = np.max(res1)
max2 = np.max(res2)
point1 = np.where(res1==max1)
point2 = np.where(res2==max2)
print(point2)
#points.append(np.where(res2==max2))
print("spend time",(cv2.getTickCount()-start)/cv2.getTickFrequency())
print(points)
cv2.circle(img,(point1[1],point1[0]),4,[0,255,0],-1)
cv2.circle(img,(point2[1],point2[0]),4,[0,255,0],-1)
rows = np.argmax(gray,axis = 0)
rowroot = np.max(rows)
colroot = np.argmax(gray[rowroot,:])
cv2.circle(img,(colroot,rowroot),4,[0,0,255],-1)
cv2.imshow("res",img)
cv2.waitKey(0)