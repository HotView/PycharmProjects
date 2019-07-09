import cv2

clicked = False
def onMouse(event,x,y,flags,param):
    global clicked
    if event== cv2.EVENT_FLAG_LBUTTON:
        clicked = True

cameraCapture = cv2.VideoCapture(0)
cv2.namedWindow("MyWindow")
cv2.setMouseCallback("MyWindow",onMouse)

print("Showing pose feed Click window or press any keys to stop.")

sucess,frame = cameraCapture.read()

<<<<<<< HEAD
while True:
    if sucess and cv2.waitKey(1)==-1 and not clicked:
        cv2.imshow('MyWindow',frame)
        sucess,frame = cameraCapture.read()
    else:
        cv2.imwrite("test.jpg", frame)
        break
=======
while sucess and cv2.waitKey(1)==-1 and not clicked:
    cv2.imshow('MyWindow',frame)
    sucess,frame = cameraCapture.read()

>>>>>>> 61326c24919029217a1e3913187e4d81184d980e
cv2.destroyWindows("MyWindow")
cameraCapture.release()