# coding=utf-8
__author__ = 'weed'

import cv2
import numpy
import utils

def recolorRC(src, dst):
    """
    BGRからRC（赤、シアン）への変換をシミュレーションする
    コードの内容：
    dst.b = dst.g = 0.5 * (src.b + src.g)
    dst.r = src.r
    :param src: BGR形式の入力画像
    :param dst: BGR形式の出力画像
    :return: None
    """
    b, g, r = cv2.split(src)
    cv2.addWeighted(b, 0.5, g, 0.5, 0, b)
    # Python: cv.AddWeighted(src1, alpha, src2, beta, gamma, dst) → None
    # Parameters:
    # src1 – first input array.
    # alpha – weight of the first array elements.
    # src2 – second input array of the same size and channel number as src1.
    # beta – weight of the second array elements.
    # dst – output array that has the same size and number of channels as the input arrays.
    # gamma – scalar added to each sum.
    # dtype – optional depth of the output array; when both input arrays have the same depth, dtype can be set to -1, which will be equivalent to src1.depth().
    # The function addWeighted calculates the weighted sum of two arrays as follows:
    # dst = src1 * alpha + src2 * beta + gamma
    cv2.merge((b, b, r), dst)

def recolorRGV(src, dst):
    """
    BGRからRGV（赤、緑、値）への変換をシミュレートする
    コードの内容：
    dst.b = min(src.b, src.g, src.r)
    dst.g = src.g
    dst.r = src.r
    :param src: BGR形式の入力画像
    :param dst: BGR形式の出力画像
    :return: None
    """
    b, g, r = cv2.split(src)
    cv2.min(b, g, b)
    cv2.min(b, r, b)
    # Python: cv2.min(src1, src2[, dst]) → dst
    # Python: cv.Min(src1, src2, dst) → None
    # Python: cv.MinS(src, value, dst) → None
    # Parameters:
    # src1 – first input array.
    # src2 – second input array of the same size and type as src1.
    # value – real scalar value.
    # dst – output array of the same size and type as src1.
    # The functions min calculate the per-element minimum of two arrays:
    #     dst = min(src1, src2)
    # or array and a scalar:
    #     dst = min(src1, value)
    # In the second variant, when the input array is multi-channel, each channel is compared with value independently.
    cv2.merge((b, g, r), dst)

def recolorCMV(src, dst):
    """
    BGRからCMV（シアン、マゼンタ、値）への変換をシミュレートする
    コードの内容：
    dst.b = max(src.b, src.g, src.r)
    dst.g = src.g
    dst.r = src.r
    :param src: BGR形式の入力画像
    :param dst: BGR形式の出力画像
    :return: None
    """
    b, g, r = cv2.split(src)
    cv2.max(b, g, b)
    cv2.max(b, r, b)
    cv2.merge((b, g, r), dst)

class VFuncFilter(object):
    """
    関数を引数にとってインスタンス化されるクラス。
    あとでapply()メソッドを使って関数を画像に適用する。
    関数は、グレースケール画像のV（値）チャンネルもしくは
    カラー画像のすべてのチャンネルに、適用される。
    """
    def __init__(self, vFunc = None, dtype = numpy.uint8):

        length = numpy.iinfo(dtype).max + 1
        # class numpy.iinfo(type)
        # 整数型の計算機の限界
        #
        # Parameters:
        # type : integer type, dtype, or instance
        # The kind of integer data type to get information about.
        #
        # >>> ii16 = np.iinfo(np.int16)
        # >>> ii16.min
        # -32768
        # >>> ii16.max
        # 32767
        # >>> ii32 = np.iinfo(np.int32)
        # >>> ii32.min
        # -2147483648
        # >>> ii32.max
        # 2147483647
        #
        # 正の整数8ビットのmaxは255なので、lengthは256になる。
        # lengthは入力データのビット数（8，16，32）によって変わる。

        self._vLookupArray = utils.createLookupArray(vFunc, length)
        # 変換用配列（256個の1次元配列）をつくる

    def apply(self, src, dst):
        # TODO: ここはわからない。後回し。
        """
        :param src: グレースケールもしくはBGR形式の入力画像
        :param dst: グレースケールもしくはBGR形式の出力画像
        :return: None
        """
        # srcFlatView = utils.createFlatView(src)
        # dstFlatView = utils.createFlatView(dst)
        # utils.applyLookupArray(self._vLookupArray, srcFlatView, dstFlatView)
        utils.applyLookupArray(self._vLookupArray, src, dst)

class VCurveFilter(VFuncFilter):
    """
    複数の(x,y)を引数にとってインスタンス化されるクラス。
    複数の(x,y)からベジエ曲線をつくり、y=f(x)の関数にする。
    あとでapply()メソッドを使って関数を画像に適用する。
    関数は、グレースケール画像のV（値）チャンネルもしくは
    カラー画像のすべてのチャンネルに、適用される。
    """
    def __init__(self, vPoints, dtype = numpy.uint8):
        VFuncFilter.__init__(self, utils.createCurveFunc(vPoints), dtype)

class TestCurveFilter(VCurveFilter):
    def __init__(self, dtype = numpy.uint8):
        VCurveFilter.__init__(self,
                              [(0,0),(35,25),(205,227),(255,255)],
                              dtype=dtype)

class BGRFuncFilter(object):
    """
    BGRチャンネルそれぞれに異なった関数を適用するフィルタ
    VFuncFilterは一つの関数しか適用できなかったが、
    このフィルタでは全体に適用する関数の他に
    BGRチャンネルそれぞれに適用する関数を使うことができる。
    """
    def __init__(self,
                 vFunc = None,
                 bFunc = None,
                 gFunc = None,
                 rFunc = None,
                 dtype = numpy.uint8):
        """
        初期化する
        :param vFunc: RGBすべてのチャンネルに適用する関数
        :type  vFunc: function
        :param bFunc: Bチャンネルに適用する関数
        :type  bFunc: function
        :param gFunc: Gチャンネルに適用する関数
        :type  gFunc: function
        :param rFunc: Rチャンネルに適用する関数
        :type  rFunc: function
        :param dtype: データタイプ。普通はunsigned int 8bit(0-255)
        :return: BGRFuncFilter
        """

        length = numpy.iinfo(dtype).max + 1
        # 正の整数8ビットのmaxは255なので、lengthは256になる。
        # lengthは入力データのビット数（8，16，32）によって変わる。

        bvFunc = utils.createCompositeFunc(bFunc, vFunc)
        self._bLookupArray = utils.createLookupArray(bvFunc, length)
        gvFunc = utils.createCompositeFunc(bFunc, vFunc)
        self._gLookupArray = utils.createLookupArray(gvFunc, length)
        rvFunc = utils.createCompositeFunc(rFunc, vFunc)
        self._rLookupArray = utils.createLookupArray(rvFunc, length)

    def apply(self, src, dst):
        """
        BGR画像にフィルタを適用する
        :param src: 入力されるBGR画像
        :param dst: 出力されるBGR画像
        :return:
        """
        b, g, r = cv2.split(src)
        utils.applyLookupArray(self._bLookupArray, b, b)
        utils.applyLookupArray(self._gLookupArray, g, g)
        utils.applyLookupArray(self._rLookupArray, r, r)
        cv2.merge([b, g, r], dst)

class BGRCurveFilter(BGRFuncFilter):
    """
    BGRチャンネルそれぞれに異なったカーブ関数を適用するフィルタ
    """
    def __init__(self,
                 vPoints = None,
                 bPoints = None,
                 gPoints = None,
                 rPoints = None,
                 dtype = numpy.uint8):
        """
        初期化する
        :param vPoints:
        :type vPoints: list[tuple]
        :param bPoints:
        :type bPoints: list[tuple]
        :param gPoints:
        :type gPoints: list[tuple]
        :param rPoints:
        :type rPoints: list[tuple]
        :param dtype:
        :return:
        """
        BGRFuncFilter.__init__(self,
                               utils.createCurveFunc(vPoints),
                               utils.createCurveFunc(bPoints),
                               utils.createCurveFunc(gPoints),
                               utils.createCurveFunc(rPoints))

class BGRPortraCurveFilter(BGRCurveFilter):
    """
    Kodak Portra Film emulation
    """
    def __init__(self, dtype = numpy.uint8):
        BGRCurveFilter.__init__(
            self,
            bPoints=[(0,0),(35,25),(205,227),(255,255)],
            gPoints=[(0,0),(27,21),(196,207),(255,255)],
            rPoints=[(0,0),(59,54),(202,210),(255,255)],
            dtype=dtype)

def strokeEdges(src, dst, blurKsize = 7, edgeKsize = 5):
    if blurKsize >= 3:
        # まずぼかし・・・
        blurredSrc = cv2.medianBlur(src, blurKsize)
        # Python: cv2.medianBlur(src, ksize[, dst]) → dst
        #
        # Parameters:
        #
        # src – input 1-, 3-, or 4-channel image;
        # when ksize is 3 or 5,
        # the image depth should be CV_8U, CV_16U, or CV_32F,
        # for larger aperture sizes, it can only be CV_8U.
        #
        # dst – destination array of the same size and type as src.
        #
        # ksize – aperture linear size;
        # it must be odd and greater than 1, for example: 3, 5, 7 ...
        #
        # The function smoothes an image
        # using the median filter with the ksize x ksize aperture.
        # Each channel of a multi-channel image is processed independently.
        # In-place operation is supported.

        graySrc = cv2.cvtColor(blurredSrc, cv2.COLOR_BGR2GRAY)
    else:
        graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # ・・・次に、エッジ検出する
    cv2.Laplacian(graySrc, cv2.cv.CV_8U, graySrc, ksize=edgeKsize)
    # 0 - 255 の整数を 0.0 - 1.0 の値にする
    normalizedInverseAlpha = (1.0 / 255) * (255 - graySrc)
    channels = cv2.split(src)
    for channel in channels:
        channel[:] = channel * normalizedInverseAlpha
    cv2.merge(channels, dst)

def applyBlur(src, dst, blurKsize = 7):
    dst[:] = cv2.medianBlur(src, blurKsize)

def applyLaplacian(src, dst, edgeSize = 5):
    graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    cv2.Laplacian(graySrc, cv2.cv.CV_8U, graySrc, ksize=edgeSize)
    dst[:] = cv2.cvtColor(graySrc, cv2.COLOR_GRAY2BGR)

def getSimpleMaskByHsv(h, s, v, hueMin, hueMax, valueMin, valueMax, sThreshold=5):

    # 色相

    _, hTargetLower = cv2.threshold(h, hueMax, 255, cv2.THRESH_BINARY_INV)
    # Python: cv2.threshold(src, thresh, maxval, type[, dst]) → retval, dst
    # Parameters:
    # src – input array (single-channel, 8-bit or 32-bit floating point).
    # dst – output array of the same size and type as src.
    # thresh – threshold value.
    # maxval – maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV thresholding types.
    # type – thresholding type (see the details below).

    # 0
    # 0
    # 0
    # thresh
    # 255
    # 255
    # 255
    # 255
    # 255

    _, hTargetHigher = cv2.threshold(h, hueMin, 255, cv2.THRESH_BINARY)
    # THRESH_BINARY
    # if src(x,y) > thresh then dst(x,y) = maxval = 255
    # if otherwise         then dst(x,y) = 0
    # 「src(x,y)がthreshより小さければ、dst(x,y)は0」

    # 255
    # 255
    # 255
    # 255
    # 255
    # thresh
    # 0
    # 0
    # 0

    hMask = cv2.bitwise_and(hTargetLower, hTargetHigher)

    # 結果

    # 0
    # 0
    # 0
    # thresh
    # 255
    # thresh
    # 0
    # 0
    # 0

    # 彩度

    # 蛍光灯の光（黄）を除外するため、
    # 極端に彩度が低く明度が高い（つまり白い）ピクセルをターゲット範囲から除外する
    # 極端に彩度が低く明度が高いところ
    _, sNotVeryLow = cv2.threshold(s, 2 ** sThreshold - 1, 255, cv2.THRESH_BINARY)
    # 彩度が31より高いならターゲット範囲（255）に入れる。
    # さもなくば非ターゲット範囲（0）。

    # 明度

    _, vTargetLower = cv2.threshold(v, valueMax, 255, cv2.THRESH_BINARY_INV)
    _, vTargetHigher = cv2.threshold(v, valueMin, 255, cv2.THRESH_BINARY)
    vMask = cv2.bitwise_and(vTargetLower, vTargetHigher)

    # 論理積をとる
    HsMask = cv2.bitwise_and(hMask, sNotVeryLow)
    HsvMask = cv2.bitwise_and(HsMask, vMask)
    return HsvMask

def letMaskMoreBright(v, mask, gamma):
    # 明度画像に+gammaのガンマ補正をかけ、明るくする
    vBrightened = cv2.addWeighted(v, 1.0-gamma/256, v, 0.0, gamma)
    # マスク範囲のみ、ガンマ補正済み明度画像を入れる
    cv2.bitwise_and(vBrightened, 255, v, mask)
    return v

def equaliseHist(src, dst):
    src = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(src)
    cv2.equalizeHist(v, v)
    cv2.merge((h, s, v), src)
    cv2.cvtColor(src, cv2.COLOR_HSV2BGR, dst)
