import cv2
import numpy as np
import time


class CaptureMannger(object):
    def __init__(self,capture,previewWindowMannger = None,shouldMirrorPreview = False):

        self.previewWindowMannger = previewWindowMannger
        self.shouldMiiorPreview = shouldMirrorPreview

        self._capture = capture
        self._channel = 0
        self._entereFrame = False
        self._frame = None
        self._imageFilename = None
        self._videoFilename = None
        self.videoEncoding = None
        self._videoWriter = None

        self._startTime = None
        self._frameElapsed = None
        self.fpsEstimate = None

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self):
        pass