# coding=utf-8
__author__ = 'weed'

import cv2
import rects
import utils

class Face(object):
    """
    顔、目、鼻、口の矩形画像データをまとめたクラス
    """
    def __init__(self):
        self.faceRect = None
        """:type : tuple"""
        self.leftEyeRect = None
        """:type : tuple"""
        self.rightEyeRect = None
        """:type : tuple"""
        self.noseRect = None
        """:type : tuple"""
        self.mouthRect = None
        """:type : tuple"""

class FaceTracker(object):
    """

    """

    def __init__(self, scaleFactor = 1.2, minNeighbors = 2,
                 flags = cv2.cv.CV_HAAR_SCALE_IMAGE):
        """

        :param scaleFactor:  各画像スケールにおける縮小量を表します
        :type  scaleFactor:  float
        :param minNeighbors: オブジェクト候補となる矩形は，最低でもこの数だけの近傍矩形を含む必要があります
        :type  minNeighbors: int
        :param flags:        このパラメータは，新しいカスケードでは利用されません
        :type  flags:        int
        :return:
        """

        self.scaleFactor = scaleFactor
        """:type : float"""
        self.minNeighbors = minNeighbors
        """:type : int"""
        self.flags = flags
        """:type : int"""

        self._faces = []
        """:type : list[Face]"""

        self._faceClassifier = cv2.CascadeClassifier(
            'cascades/haarcascade_frontalface_alt.xml')
        """:type : cv2.CascadeClassifier"""
        self._eyeClassifier = cv2.CascadeClassifier(
            'cascades/haarcascade_eye.xml')
            # 'cascades/haarcascade_eye_tree_eyeglasses.xml')

        """:type : cv2.CascadeClassifier"""
        self._noseClassifier = cv2.CascadeClassifier(
            'cascades/haarcascade_mcs_nose.xml')
        """:type : cv2.CascadeClassifier"""
        self._mouthClassifier = cv2.CascadeClassifier(
            'cascades/haarcascade_mcs_mouth.xml')
        """:type : cv2.CascadeClassifier"""

    @property
    def faces(self):
        """
        :return: 追跡している顔の特徴
        :rtype:  list[Face]
        """
        return self._faces

    ####### 初期化ここまで #######

    def _detectOneObject(self, classifier, image, rect,
                         imageSizeToMinSizeRatio):
        """
        探索領域の中で目や鼻や口を探す
        :param classifier: 使う分類器
        :type  classifier: cv2.CascadeClassifier
        :param image:      全体画像
        :param rect:       対象を探す領域
        :param imageSizeToMinSizeRatio: 無視するサイズの割合
        :return: 対象の(x, y, w, h)
        :rtype : tuple
        """
        x, y, w, h = rect

        minSize = utils.widthHeightDividedBy(image, imageSizeToMinSizeRatio)

        subImage = image[y:y+h, x:x+w]
        """:type : numpy.ndarray"""

        subRects = classifier.detectMultiScale(
            subImage, self.scaleFactor, self.minNeighbors,
            self.flags, minSize
        )
        """:type : list[tuple]"""

        if len(subRects) == 0:
            return None

        subX, subY, subW, subH = subRects[0]
        return (x+subX, y+subY, subW, subH)

    def update(self, image):
        """

        :param image:
        :return:
        """
        self._faces = []

        if utils.isGray(image):
            image = cv2.equalizeHist(image)
            # The algorithm normalizes the brightness and increases the contrast of the image.
            # 明るさをノーマライズしコントラストを強める
        else:
            # グレースケール画像に変換する
            image = cv2.cvtColor(image, cv2.cv.CV_BGR2GRAY)
            image = cv2.equalizeHist(image)

        # まず顔を検出する

        # 画像の1/8 * 1/8 = 1/64より小さい顔は認識しない
        minSize = utils.widthHeightDividedBy(image, 8) # => (w, h)

        faceRects = self._faceClassifier.detectMultiScale(
            image, self.scaleFactor, self.minNeighbors, self.flags, minSize
        )
        """:type : list[tuple]"""
        # => (x, y, w, h)
        #
        # void CascadeClassifier::detectMultiScale(
        #   const Mat& image, vector<Rect>& objects, double scaleFactor=1.1, int minNeighbors=3,
        #   int flags=0, Size minSize=Size()
        # )
        # 入力画像中から異なるサイズのオブジェクトを検出します．検出されたオブジェクトは，矩形のリストとして返されます．
        #
        # パラメタ:
        # image – CV_8U 型の行列．ここに格納されている画像中からオブジェクトが検出されます
        # objects – 矩形を要素とするベクトル．それぞれの矩形は，検出したオブジェクトを含みます
        # scaleFactor – 各画像スケールにおける縮小量を表します
        # minNeighbors – オブジェクト候補となる矩形は，最低でもこの数だけの近傍矩形を含む必要があります
        # flags – このパラメータは，新しいカスケードでは利用されません．
        #   古いカスケードに対しては，cvHaarDetectObjects 関数の場合と同じ意味を持ちます
        # minSize – オブジェクトが取り得る最小サイズ．これよりも小さいオブジェクトは無視されます

        # 顔を検出できたら・・・
        if faceRects is not None:
            # 顔ごとに・・・
            for faceRect in faceRects:

                # faceオブジェクトをつくる
                face = Face()
                face.faceRect = faceRect

                x, y, w, h = faceRect

                # 左目を探す
                searchRect = (x+w/7, y, w*2/7, h/2)
                # □■■□□□□
                # □■■□□□□
                # □□□□□□□
                # □□□□□□□
                face.leftEyeRect = self._detectOneObject(
                    self._eyeClassifier, image, searchRect, 64
                )
                # 1/64 * 1/64より小さいオブジェクトを無視する
                # if face.leftEyeRect == None:
                #     print("Left Eye detect failed")

                # 右目を探す
                searchRect = (x+w*4/7, y, w*2/7, h/2)
                # □□□□■■□
                # □□□□■■□
                # □□□□□□□
                # □□□□□□□
                face.rightEyeRect = self._detectOneObject(
                    self._eyeClassifier, image, searchRect, 64
                )

                # 鼻を探す
                searchRect = (x+w/4, y+h/4, w/2, h/2)
                # □□□□□□□
                # □□■■■□□
                # □□■■■□□
                # □□□□□□□
                face.noseRect = self._detectOneObject(
                    self._noseClassifier, image, searchRect, 32
                )
                # 1/32 * 1/32より小さいオブジェクトを無視する

                # 口を探す
                searchRect = (x+w/6, y+h*2/3, w*2/3, h/3)
                # □□□□□□□
                # □□□□□□□
                # □■■■■■□
                # □■■■■■□
                face.mouthRect = self._detectOneObject(
                    self._mouthClassifier, image, searchRect, 16
                )
                # 1/16 * 1/16より小さいオブジェクトを無視する

                self._faces.append(face)

    def drawDebugRects(self, image):
        """

        :param image: 全体画像
        :return: None
        """

        if utils.isGray(image):
            faceColor     = 255
            leftEyeColor  = 255
            rightEyeColor = 255
            noseColor     = 255
            mouthColor    = 255
        else:
            faceColor     = (255, 255, 255) # 白
            leftEyeColor  = (  0,   0, 255) # 赤
            rightEyeColor = (  0, 255, 255) # 黄
            noseColor     = (  0, 255,   0) # 緑
            mouthColor    = (255,   0,   0) # 青

        for face in self.faces:
            # rects.outlineRect(image, face.faceRect    , faceColor    )
            # rects.outlineRect(image, face.leftEyeRect , leftEyeColor )
            # rects.outlineRect(image, face.rightEyeRect, rightEyeColor)
            # rects.outlineRect(image, face.noseRect    , noseColor    )
            # rects.outlineRect(image, face.mouthRect   , mouthColor   )
            rects.outlineRectWithTitle(image, face.faceRect    , faceColor    , 'Face')
            rects.outlineRectWithTitle(image, face.leftEyeRect , leftEyeColor , 'Left Eye')
            rects.outlineRectWithTitle(image, face.rightEyeRect, rightEyeColor, 'Right Eye')
            rects.outlineRectWithTitle(image, face.noseRect    , noseColor    , 'Nose')
            rects.outlineRectWithTitle(image, face.mouthRect   , mouthColor   , 'Mouth')


if __name__=="__main__":
    FaceTracker()