import cv2
from managers import WindowManager,CaptureManager
import filters
class Cameo(object):
    def __init__(self):
        self._windowManager  = WindowManager('Cameo',self.onKeypress)

        self._captureManager = CaptureManager(cv2.VideoCapture(0),self._windowManager,False)
        self._curveFilter = filters.BlurFilter()
    def run(self):
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame

            filters.strokeEdges(frame,frame)
            self._curveFilter.apply(frame,frame)
            fps = self._captureManager.getFps()
            self._captureManager.showFps(frame,(20,30),str(fps),1,2)

            self._captureManager.exitFrame()
            self._windowManager.processEvents()#每次循环都要进行一次事件的扫描处理。
    def onKeypress(self,keycode):
        """sapce -> take a screeshot
           tab -> start/stop recording a screencast
           escap -.Quit.
        """
        if keycode == 32:#sapce
            self._captureManager.writeImage("screenshot.png")

        elif keycode== 9:#tab
            print("tab press")
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo("screencast.avi")
                print("starting writing video")
            else:
                self._captureManager.stopWritingVideo()
                print("stop writing video")
        elif keycode== 27:
            self._windowManager.destroyWindows()
if __name__ == '__main__':
    Cameo().run()