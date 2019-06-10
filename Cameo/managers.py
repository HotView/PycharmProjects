import cv2
import numpy
import time
import matplotlib.pyplot as plt
"""提取图像流"""
class CaptureManager(object):

    def __init__(self,capture,previewWindowManager = None,shouldMirrorPreview = False):
        self.previewWindowManager = previewWindowManager
        self.shouldMirrorPreview = shouldMirrorPreview

        self._capture = capture
        self._channel = 0
        self._enteredFrame  = False
        self._frame = None
        self._imageFilename = None
        self._videoFilename = None
        self._videoEncoding  = None
        self._videoWriter = None

        self._startTime = None
        self._frameElapsed = int(0)
        self._fpsEstimate = None


    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self,value):
        if self._channel != value:
            self._channel = value
            self._frame = None
    @property
    def frame(self):
        if self._enteredFrame and self._frame is None:
            _,self._frame = self._capture.retrieve()
        return self._frame

    @property
    def isWritingImage(self):
        return self._imageFilename is not None

    @property
    def isWritingVideo(self):
        return self._videoFilename is not None



    def enterFrame(self):
        '''Capture the next frame,if any.'''
        assert not self._enteredFrame,'previous enterFrame() had no matching exitFrame()'

        if self._capture is not None:
            self._enteredFrame = self._capture.grab()

    def exitFrame(self):

        if self.frame is None:
            self._enteredFrame = False
            return
        if self._frameElapsed==0:
            self._startTime = time.time()
        else:
            timeElapsed = time.time()-self._startTime
            self._fpsEstimate = self._frameElapsed/timeElapsed
        self._frameElapsed+=1
        """Draw to the window,if any """
        if self.previewWindowManager is not None:
            if self.shouldMirrorPreview:
                mirroredFrame  = numpy.fliplr(self._frame).copy()
                self.previewWindowManager.show(mirroredFrame)
            else:
                self.previewWindowManager.show(self._frame)
        """Write to the image file ,if any"""
        if self.isWritingImage:
            cv2.imwrite(self._imageFilename,self._frame)
            self._imageFilename = None

        self._writeVideoFrame()
        self._frame =None
        self._enteredFrame = False
    """writeImage,startWritingVideo,stopWritingVideo是共有函数，它们简单的记录了文件写入操作的函数，
    然而实际的写入操作会推迟下一次调用exitFrame（）函数"""
    def writeImage(self,filename):
        self._imageFilename = filename

    def startWritingVideo(self,filename,encoding = cv2.VideoWriter_fourcc('I','4','2','0')):
        self._videoFilename = filename
        self._videoEncoding = encoding
    def stopWritingVideo(self):
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter = None

    def _writeVideoFrame(self):
        print("调用writeVideoFrame函数")
        if not self.isWritingVideo:
            print("not isWritingVideo")
            return
        if self._videoWriter is None:
            fps = self._capture.get(cv2.CAP_PROP_FPS)
            if fps == 0.0:
                if self._frameElapsed<20:
                    return
                else:
                    fps =  self._fpsEstimate
            size = (int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self._videoWriter = cv2.VideoWriter(self._videoFilename,self._videoEncoding,fps,size)
            print("VideoWriter")
        self._videoWriter.write(self._frame)
        print("################################")

    def showFps(self, img, size, text, font, font_width):
        cv2.putText(img, text, size, cv2.FONT_HERSHEY_SIMPLEX, font, (0, 255, 0), font_width)
    def getFps(self):
        if self._fpsEstimate:
            return self._fpsEstimate
        else:
            return 0
        """plt.figure()
        plt.imshow(self._frame)
        plt.show()"""
"""实现仅仅支持键盘事件，这对于Cameo足够了，若要支持鼠标事件可修改WindowManager。
   例如，可以扩展类的接口使其包括mouseCallback属性，而其他属性不变。
   对于其他非OpenCV的事件框架，可以像添加callback属性一样支持其他事件类型。
   
   也可以通过Pygame（而不是通过OpenCV）的窗口处理和事件框架功能实现。
   
   正确处理退出事件，使得WindowManager基类的实现有所提高，例如用户单击窗口就可以实现退出。
   事实上，Pygame也可以处理许多其他的事件类型。
"""
class WindowManager(object):
    def __init__(self,windowName,keypressCallback = None):
        self.keypressCallback = keypressCallback

        self._windowName = windowName
        self._isWindowCreated = False

    @property
    def isWindowCreated(self):

        return  self._isWindowCreated
    def createWindow(self):
        cv2.namedWindow(self._windowName)
        self._isWindowCreated = True
    def show(self,frame):
        cv2.imshow(self._windowName,frame)

    def destroyWindows(self):
        cv2.destroyWindow(self._windowName)
        self._isWindowCreated  = False
    def processEvents(self):
        keycode = cv2.waitKey(1)

        if self.keypressCallback is not None and keycode != -1:
            keycode &= 0xFF
            self.keypressCallback(keycode)