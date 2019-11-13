import cv2
cameraCapture = cv2.VideoCapture(0)
fps = 20
size = (int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
videoWrite = cv2.VideoWriter(
    '../camera/video/myvideo.avi',cv2.VideoWriter_fourcc("I","4","2","0"),
    fps,size)
success,frame = cameraCapture.read()
numFramesRemaining = 10*fps-1
while success and numFramesRemaining>0:
    videoWrite.write(frame)
    filename = '../camera/image/'+str(numFramesRemaining)+'.png'
    cv2.imwrite(filename,frame)
    success,frame = cameraCapture.read()
    numFramesRemaining-=1
cameraCapture.release()