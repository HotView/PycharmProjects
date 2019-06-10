import cv2
import numpy as np

img = cv2.imread("target.jpg")
pos_fist = [0,0]
pos_second = [0,0]
def on_mouse(event,x,y,flags,param):
    print(flags,"flags")
    print(param,"param")
    if event== cv2.EVENT_MOUSEMOVE:
        postext = "pos"+str(x)+","+str(y)
        cv2.putText(img,postext,(30,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0))
        cv2.putText(img, postext, (pos_fist[0], pos_fist[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        cv2.imshow("target", img)
        pos_prev = postext
        print(x,y)
    elif event == cv2.EVENT_LBUTTONDOWN:
        pos_fist[0] = x
        pos_fist[1] = y
        postext = "pos" + str(x) + "," + str(y)
        cv2.putText(img, postext, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        cv2.imshow("target", img)
    elif event == cv2.EVENT_LBUTTONUP:
        postext = "pos" + str(x) + "," + str(y)
        cv2.putText(img, postext, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        cv2.imshow("target", img)
cv2.namedWindow("target",1)
cv2.setMouseCallback("target",on_mouse)
while True:
    img = cv2.imread("target.jpg")
    cv2.waitKey(10)
    print("#")
