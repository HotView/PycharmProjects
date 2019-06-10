import cv2
def bar(pos):
    pass
def onmouse(event,x,y,flags,param):
    if event== cv2.EVENT_LBUTTONDOWN:
        print(cv2.getTrackbarPos("bar","img"))
print("fjdksfsdk")
cv2.namedWindow("img",0)
param = [21,14,2,21]
pos= 1
cv2.createTrackbar("bar","img",0,255,bar)
#cv2.imshow()
pos = cv2.getTrackbarPos("bar","img")
cv2.setMouseCallback("img",onmouse)
cv2.waitKey(0)

