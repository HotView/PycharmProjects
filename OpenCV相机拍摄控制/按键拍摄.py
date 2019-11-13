import cv2
cameraCapture = cv2.VideoCapture(0)
fps = 20
size = (int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
success,frame = cameraCapture.read()
i = 0
cv2.namedWindow("win")
while success:
    cv2.imshow("win",frame)
    if cv2.waitKey(200)==32:
        filename = '../camera/image/'+str(i)+'.png'
        cv2.imwrite(filename,frame)
        i= i+1
    elif cv2.waitKey(100)==27:
        break
    success,frame = cameraCapture.read()
cameraCapture.release()