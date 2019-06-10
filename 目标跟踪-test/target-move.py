import cv2
import numpy as np
cap = cv2.VideoCapture(0)
ret,frame = cap.read()
r,h,c,w = 250, 90, 400, 125
track_window = (c,r,w,h)
img = cv2.imread("target.jpg")
print(img.shape)
roi = img[400:,0:180]
hsv_roi = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi,np.array((0.,60.,32.)),np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
# Setup the termination criteria,either 10 iteration or move by atleast 1 pt
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,1)

while(1):
    ret,frame = cap.read()
    if ret is True:
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # apply meanshift to get the new location
        ret,track_window = cv2.meanShift(dst,track_window,term_crit)

        # Draw it on image
        x,y,w,h = track_window
        img2 = cv2.rectangle(frame,(x,y),(x + w,y + h),255,2)
        cv2.imshow("img2",img2)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k)+'.jpg',img2)
    else:
        break
cv2.destroyAllWindows()
cap.release()
