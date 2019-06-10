import cv2
cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    cv2.imwrite("target.jpg",frame)
    cv2.imshow("img",frame)
    k = cv2.waitKey(10) & 0xff
    if k == 'q':
        break
