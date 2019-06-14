import cv2
img = cv2.imread("luosi.jpg")
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
a = cv2.connectedComponentsWithStats(img_gray)
for x in a:
    print(x)
centroids = a[3]
for point in centroids:
    x,y = map(int,point)
    cv2.circle(img,(x,y),3,(0,255,0))
cv2.imshow("img",img)
cv2.waitKey(0)