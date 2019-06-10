import numpy as np
import cv2
import os
import sys
cap = cv2.VideoCapture(0)
ret,frame = cap.read()
r,h,c,w = 250,90,400,125
track_window = (c,r,w,h)
#set initial location of window
roi = frame[r:r+h,c:c+w]
hsv_roi = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi,np.array((0.,60.,32.)),np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
#setup the termination criteria
term_crit = (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,10,1)

while(1):
    ret,frame = cap.read()

    if ret==True:
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)


        #apply meanchsift to get the new locayion
        ret,track_window =cv2.meanShift(dst,track_window,term_crit)
        #draw it on image
        x,y,w,h = track_window
        img2 = cv2.rectangle(frame,(x,y),(x+w,y+h),255,2)
        cv2.imshow('img2',img2)

        k = cv2.waitKey(60)&0xff
        if k==27:
            break
        else:
            dir  = os.getcwd()+"\\"+"test\\"+str(k)+".jpg"
            cv2.imwrite(dir,img2)

    else:
        break
cv2.destroyAllWindows()
cap.release()