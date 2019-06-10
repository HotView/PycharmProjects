import cv2
import time
clicked = False
def onMouse(event,x,y,flags,param):
    global clicked
    if event==cv2.EVENT_FLAG_LBUTTON:
        clicked = True
cameraCaptures = cv2.VideoCapture(0)
fps = 30 #  an assumption
size = (int(cameraCaptures.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cameraCaptures.get(cv2.CAP_PROP_FRAME_HEIGHT)))
videoWriter = cv2.VideoWriter("MyOutputVid.avi",cv2.VideoWriter_fourcc('I','4','2','0'),fps,size)
## 显示video
cv2.namedWindow("Video")
cv2.setMouseCallback('Video',onMouse)
success,frame = cameraCaptures.read()
numFrameRemaining =10*fps-1

while success and numFrameRemaining>0  and not clicked :
    start = time.time()
    videoWriter.write(frame)
    #cv2.imshow('Video',frame)
    success,frame =cameraCaptures.read()
    numFrameRemaining-=1
    end  =time.time()
    cv2.imshow('Video',frame)
    print(end-start)
    print(end)

cv2.destroyWindow("Video")
cameraCaptures.release()