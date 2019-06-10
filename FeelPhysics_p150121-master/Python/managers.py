# coding=utf-8

import cv2
import numpy
import time
import pygame
from pygame.locals import *
import utils

class CaptureManager(object):

    def __init__(self, capture, previewWindowManager = None,
                 shouldMirrorPreview = False, scaleRatio = 1.0):
        """
        コンストラクタ
        :param capture: cv2.VideoCapture
        :param previewWindowManager: WindowManager
        :param shouldMirrorPreview: bool
        :return: CaptureManager
        """

        self.previewWindowManager = previewWindowManager
        """:type : WindowManager"""
        self.shouldMirrorPreview  = shouldMirrorPreview
        """:type : bool"""

        self._capture       = capture
        """:type : cv2.VideoCapture"""
        self._channel       = 0
        """:type : int"""
        self._enteredFrame  = False
        """:type : bool"""
        self._frame         = None
        """:type : numpy.ndarray"""
        self._imageFilename = None
        """:type : str"""
        self._videoFilename = None
        """:type : str"""
        self._videoEncoding = None
        """:type : int"""
        self._videoWriter   = None
        """:type : cv2.VideoWriter"""

        self._startTime     = None
        """:type : float"""
        self._framesElapsed = long(0)
        """:type : long"""
        self._fpsEstimate   = None
        """:type : float"""

        self.paused        = False
        self._pausedFrame  = None

        self._scaleRatio   = scaleRatio

    @property
    def channel(self):
        """
        チャンネル数を返す
        :return: int
        """
        return self._channel

    @channel.setter
    def channel(self, value):
        """
        チャンネル数を入力する
        :param value: int
        :return:
        """
        if self._channel != value:
            self._channel = value
            self._frame   = None

    @property
    def frame(self):
        """
        キャプチャから取ったフレームをデコードしてself._frameに入れて、それを返す
        exitFrame()の最初に呼ばれる
        :return: numpy.ndarray
        """

        # 新しいフレームに入ったが新しいフレームができていないとき、
        # もしくは、一時停止に入る瞬間
        # もしくは、一時停止から抜ける瞬間
        if self._enteredFrame and self._frame is None and \
                (self._pausedFrame is None or self.paused is False):
            # VideoCaptureから取ったフレームをデコードする
            _, self._frame = self._capture.retrieve(
                channel = self.channel
            )
            # VideoCapture::retrieve
            # Decodes and returns the grabbed video frame.
            # → retval, image

        # 一時停止しているとき
        if self.paused is True:
            # 一時停止した瞬間
            if self._pausedFrame is None:
                # 現在のフレームを一時停止フレームに保存する
                # self._pausedFrame = self._frame
                self._pausedFrame = self._frame
                print(id(self._pausedFrame))
                print(id(self._frame))
            else:
                # 一時停止フレームを返す

                # self._frame[:] = self._pausedFrame
                # エラーが出るコード。左辺がNoneTypeなので代入できない

                # ディープコピー（idを新たに起こす）する
                self._frame = self._pausedFrame.copy()
        # 一時停止していないとき一時停止フレームが残っていたら・・・
        elif self._pausedFrame is not None:
            # 削除する
            self._pausedFrame = None

        # 画像サイズに縮尺倍率をかける
        height = int(self._frame.shape[0] * self._scaleRatio)
        width  = int(self._frame.shape[1] * self._scaleRatio)
        self._frame = cv2.resize(self._frame[:], (width, height))

        return self._frame

    @property
    def isWritingImage(self):
        """
        画像ファイルに書き出し中か
        :return: bool
        """
        return self._imageFilename is not None

    @property
    def isWritingVideo(self):
        """
        動画ファイルに書き出し中か
        :return: bool
        """
        return self._videoFilename is not None

    ########## ここまでがクラスのセットアップ ############

    def enterFrame(self):
        """
        とにかく、次のフレームをキャプチャーする
        :return:
        """

        # 最初に直前のフレームが解放されているか確認する
        assert not self._enteredFrame, \
            '直前のenterFrame()はexitFrame()していません'

        if self._capture is not None: # cv2.VideoCaptureがまだあれば・・・
            self._enteredFrame = self._capture.grab()
            # VideoCapture::grab
            # Grabs the next frame from video file or capturing device.

    def exitFrame(self):
        """
        ウィンドウに描画する。ファイルに書き出す。フレームを解放する。
        :return:
        """

        # 取得したフレームがデコード可能かを確認する
        # self.frameメソッドを呼ぶことでgrabしたVideoCaptureをデコードしている
        # これがNoneだとreturnでプログラムを終了させる
        # TODO: 意味不明「ゲッターはフレームを回収してキャッシュするかもしれない」
        # if self.frame is None:
        #     self._enteredFrame = False
        #     return

        # FPS測定値と関係する変数を更新する
        # 測定開始
        if self._framesElapsed == 0:
            self._startTime = time.time()
        # 測定中
        else:
            timeElapsed = time.time() - self._startTime
            """:type : float"""
            # FPS = 表示したフレームの枚数 / 秒
            self._fpsEstimate = self._framesElapsed / timeElapsed
        self._framesElapsed += 1


        # 画像スケールを2倍に上げる
        # self._frame = cv2.resize(self._frame[:], (1280, 720))


        #とにかく、ウィンドウに描画する
        if self.previewWindowManager is not None:
            if self.shouldMirrorPreview:
                mirroredFrame = numpy.fliplr(self._frame).copy()
                self.previewWindowManager.show(mirroredFrame)
            else:
                self.previewWindowManager.show(self._frame)

          # とにかく、画像ファイルに書き出す
        if self.isWritingImage:
            cv2.imwrite(self._imageFilename, self._frame)
            self._imageFilename = None # 書き出し終わったら解放する

        # とにかく、動画ファイルに書き出す
        if self.isWritingVideo:
            self._writeVideoFrame()

        # フレームを解放する
        self._frame = None
        self._enteredFrame = False

    def writeImage(self, filename):
        """
        次のフレームを画像ファイルに書き出す
        :param filename: str
        :return:
        """
        self._imageFilename = filename

    def startWritingVideo(self, filename,
                          encoding=cv2.cv.CV_FOURCC('I','4','2','0')):
        """
        解放したフレームを動画ファイルに書き出すのを始める
        :param filename: str
        :param encoding: int
        :return:
        """
        self._videoFilename = filename
        self._videoEncoding = encoding

    def stopWritingVideo(self):
        """
        解放したフレームを動画ファイルに書き出すのを止める
        :return:
        """
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter   = None

    def _writeVideoFrame(self):

        # 動画書き出し中でなければ中止する
        if not self.isWritingVideo:
            return

        # _videoWriterが初期化されていなければ、初期化する
        if self._videoWriter is None:
            fps = self._capture.get(cv2.cv.CV_CAP_PROP_FPS)
            """:type : float"""
            if fps == 0.0:
                # cv2.VideoCaptureのFPSがわからないので、測定する
                # 20フレームまで待つ
                if self._framesElapsed < 20:
                    return
                else:
                    fps = self._fpsEstimate
            # cv2.VideoWriterを初期化する際に、画面サイズに縮尺倍率をかける
            size = (
                int(self._capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH ) * self._scaleRatio),
                int(self._capture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT) * self._scaleRatio)
            )
            # _videoWriterを初期化する
            self._videoWriter = cv2.VideoWriter(
                self._videoFilename, self._videoEncoding, fps, size
            )
            """:type : cv2.VideoWriter"""

        self._videoWriter.write(self._frame)

class WindowManager(object):

    def __init__(self, windowName, keypressCallback = None):
        self.keypressCallback = keypressCallback
        """:type : function"""

        self._windowName      = windowName
        """:type : str"""
        self._isWindowCreated = False
        """:type : bool"""

    @property
    def isWindowCreated(self):
        """
        ウィンドウが存在するかを返す
        :return: bool
        """
        return self._isWindowCreated

    def createWindow(self):
        """
        ウィンドウをつくる
        :return:
        """
        cv2.namedWindow(self._windowName)
        self._isWindowCreated = True

    def show(self, frame):
        """
        ウィンドウにフレームを表示する
        :param frame: numpy.ndarray
        :return:
        """
        cv2.imshow(self._windowName, frame)

    def destroyWindow(self):
        """
        ウィンドウを消す
        :return:
        """
        cv2.destroyWindow(self._windowName)
        self._isWindowCreated = False

    def processEvents(self):
        keycode = cv2.waitKey(1)
        """:type : int"""
        if self.keypressCallback is not None and keycode != -1:
            # GTKによってエンコードされた非ASCII情報を捨てる
            keycode &= 0xFF
            self.keypressCallback(keycode)

class PygameWindowManager(WindowManager):
    def createWindow(self):
        pygame.display.init()
        pygame.display.set_caption(self._windowName)
        self._isWindowCreated = True
    def show(self, frame):
        # Find the frame's dimensions in (w, h) format
        frameSize = frame.shape[1::-1]
        # Convert the frame to RGB, which Pygame requires.
        # if utils.isGray(frame):
        #     conversionType = cv2.COLOR_GRAY2BGR
        # else:
        conversionType = cv2.COLOR_BGR2RGB
        rgbFrame = cv2.cvtColor(frame, conversionType)
        # Convert the frame to Pygame's Surface type.
        pygameFrame = pygame.image.frombuffer(
            rgbFrame.tostring(), frameSize, 'RGB'
        )
        # pygameFrame = pygame.transform.scale(pygameFrame, (1280, 720))
        # Resize the window to match the frame.
        displaySurface = pygame.display.set_mode(frameSize)
        # displaySurface = pygame.display.set_mode((1280, 720), RESIZABLE)
        # Blit and display the frame.
        displaySurface.blit(pygameFrame, (0, 0))
        pygame.display.flip()
        # pygame.display.set_mode((640, 360), FULLSCREEN)
    def destroyWindow(self):
        pygame.display.quit()
        self._isWindowCreated = False
    def processEvents(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and \
                self.keypressCallback is not None:
                self.keypressCallback(event.key)
            elif event.type == pygame.QUIT:
                self.destroyWindow()
                return
