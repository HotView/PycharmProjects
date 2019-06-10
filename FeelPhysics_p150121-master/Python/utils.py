# coding=utf-8
__author__ = 'weed'

import cv2
import numpy
import math

WHITE         = (255,255,255)
RED           = (  0,  0,255)
GREEN         = (  0,128,  0)
BLUE          = (255,  0,  0)
SKY_BLUE      = (255,128,128)
BLACK         = (  0,  0,  0)
DARKSLATEGRAY = ( 79, 79, 47)
TEAL          = (128,128,  0)

def getVelocityVector(positionHistory, population=1, numFramesDelay=0):
    # populationは母集団。すなわち、何フレーム分の位置データを用いて速度を求めるか。
    # populationが4、numFramesDelayが6の場合は
    # vはPt[-1-6-2=-9],Pt[-1-6+2=-5]を参照する。

    # indexPtBegin = -1-numFramesDelay-int(population/2)  # ptBegin: 始点
    # indexPtEnd   = -1-numFramesDelay+int(population/2)  # ptEnd  : 終点
    indexPtBegin = -1-numFramesDelay             # ptBegin: 始点
    indexPtEnd   = -1-numFramesDelay+population  # ptEnd  : 終点

    # 追跡開始直後
    if len(positionHistory) < -indexPtBegin \
            or positionHistory[indexPtBegin] is None \
            or positionHistory[indexPtEnd]   is None:
        return None
    else:
        ptBeginNp = numpy.array(positionHistory[indexPtBegin])
        ptEndNp   = numpy.array(positionHistory[indexPtEnd]  )
        # 移動ベクトル Δpt = ptEnd - ptBegin
        deltaPtNp = ptEndNp - ptBeginNp

        # 移動してなければNoneを返す
        notMoved = (deltaPtNp == numpy.array([0,0]))
        if notMoved.all():
            return None
        # 移動していれば、速度ベクトル = 移動ベクトル / 母数
        else:
            velocityVectorNp = deltaPtNp / float(population)
            velocityVector   = tuple(velocityVectorNp)
            return velocityVector

def getAccelerationVector2(positionHistory, populationVelocity, populationAcceleration=12,
                           numFramesDelay=0, coAcceleration=200):

    velocityVectorEnd   = getVelocityVector(positionHistory, populationVelocity,
                                              populationVelocity)
    # indexPtBegin = -1-popV = -1-popV  # ptBegin: 始点
    # indexPtEnd   = -1+popV-popV = -1  # ptEnd  : 終点
    velocityVectorBegin = getVelocityVector(positionHistory, populationVelocity,
                                              populationVelocity+populationAcceleration)
    # indexPtBegin = -1     -(popV+popAcl) = -1-popV-popAcl  # ptBegin: 始点
    # indexPtEnd   = -1+popV-(popV+popAcl) = -1     -popAcl  # ptEnd  : 終点

    if velocityVectorBegin is None or velocityVectorEnd is None:
        return None
    else:
        velocityVectorBeginNp = numpy.array(velocityVectorBegin)
        velocityVectorEndNp   = numpy.array(velocityVectorEnd)
        # 移動ベクトル ΔVelocityVector = velocityVectorEnd - velocityVectorBegin
        deltaVelocityVectorNp = velocityVectorEndNp - velocityVectorBeginNp

        # 変化してなければNoneを返す
        notChanged = (deltaVelocityVectorNp == numpy.array([0,0]))
        if notChanged.all():
            return None
        # 移動していれば、速度ベクトル = 移動ベクトル * 係数 / 母数
        else:
            accelerationVectorNp = deltaVelocityVectorNp * coAcceleration / float(populationAcceleration)
            accelerationVector   = tuple(accelerationVectorNp)
            return accelerationVector

def getAccelerationVector(positionHistory, population=2, numFramesDelay=0):
    pop = int(population / 2)  # 切り捨て
    if len(positionHistory) < 1+2*pop+numFramesDelay \
            or positionHistory[-1-numFramesDelay] is None \
            or positionHistory[-1-pop-numFramesDelay] is None \
            or positionHistory[-1-2*pop-numFramesDelay] is None:
        return None
    else:
        # [-1-pop]から[-1-2*pop]のときの速度
        velocity0 = getVelocityVector(positionHistory, pop, pop+numFramesDelay)
        # [-1]から[-1-pop]のときの速度
        velocity1 = getVelocityVector(positionHistory, pop, numFramesDelay)
        if velocity0 is not None and velocity1 is not None:

            printVector('v0', velocity0)
            printVector('v1', velocity1)

            v0np = numpy.array(velocity0)
            v1np = numpy.array(velocity1)
            dvnp = v1np - v0np  # v1 - v0 = Δv
            # 速度変化してなければNoneを返す
            areSameVelocity_array = (dvnp == numpy.array([0,0]))
            if areSameVelocity_array.all():
                return None
            else:
                dvnp = dvnp * 10.0 / pop
                vector = tuple(dvnp)

                printVector('a ', vector)

                return vector

def getAccelerationVectorStartStop(
        positionHistory,
        population=6,
        numFramesDelay=3,
        coForceVectorStrength=25.0):

    ### 静止判定

    # v6 - v3 = Δv3 = a3
    #
    v6 = getVelocityVector(positionHistory, 3, 0+numFramesDelay)
    v3 = getVelocityVector(positionHistory, 3, 3+numFramesDelay)

    v6np = numpy.array([0,0]) if v6 is None else numpy.array(v6)
    v3np = numpy.array([0,0]) if v3 is None else numpy.array(v3)

    v6size = math.sqrt(v6np[0]**2 + v6np[1]**2)
    v3size = math.sqrt(v3np[0]**2 + v3np[1]**2)

    if 20 < math.fabs(v6size - v3size) and (v6size < 2.0 or v3size < 2.0):
        # print '静止／急発進した ' + str(int(vSizeAfter - vSizeBefore))
        a3np = (v6np - v3np) * coForceVectorStrength / 3
        # 加速度が0ならNoneを返す
        areSameVelocity_array = (a3np == numpy.array([0,0]))
        if areSameVelocity_array.all():
            return None
        else:
            vector = tuple(a3np)
            return 'quickMotion', vector
    else:
        return 'usual'

def getAccelerationVectorFirFilter(
        positionHistory,
        population=6,
        numFramesDelay=3,
        coForceVectorStrength=25.0):
    # populationVelocityは6
    # v_6 - v_0 = Δv0 = a_0
    v11 = getVelocityVector(positionHistory, 6, numFramesDelay)
    v10 = getVelocityVector(positionHistory, 6, population+numFramesDelay)
    if v11 is None or v10 is None:
        pass
    else:
        v11np = numpy.array(v11)
        v10np = numpy.array(v10)
        anp = (v11np - v10np) * coForceVectorStrength / population
        # 加速度が0ならNoneを返す
        areSameVelocity_array = (anp == numpy.array([0,0]))
        if areSameVelocity_array.all():
            return None
        else:
            vector = tuple(anp)
            return vector

def printVector(name, tuple):
    tupleInt = (int(tuple[0]), int(tuple[1]))
    # print name + ': ' + str(tupleInt)

def getAccelerationVectorVelocitySensitive(positionHistory):
    # positionHistory[-6]とpositionHistory[-7]の
    # あいだの距離が40ピクセル以上のときは母数2で加速度を求める
    vVector = getVelocityVector(positionHistory, 1, 5)
    if vVector is None:
        pass
    elif 40 < math.sqrt(vVector[0]**2 + vVector[1]**2):
        # print '40 < v'
        return getAccelerationVector(positionHistory, 6, 3)
    else:
        return getAccelerationVector(positionHistory, 12, 0)

def cvArrow(img, pt, vector, lengthTimes, color, thickness=1, lineType=8, shift=0):
    if int(vector[0]) == 0 and int(vector[1]) == 0:
        pass
    else:
        cvArrowBase(img, pt, vector, lengthTimes, WHITE, thickness+2, lineType=8, shift=0)
        cvArrowBase(img, pt, vector, lengthTimes, color, thickness, lineType=8, shift=0)

def cvArrowBase(img, pt, vector, lengthTimes, color, thickness=1, lineType=8, shift=0):
    """
    矢印を描画する
    :param img: フレーム
    :param pt: 起点（タプル）
    :param vector: ベクトル（タプル）
    :param lengthTimes: 矢印の長さの倍率
    :param color: 色
    :param thickness: 太さ
    :param lineType: ？
    :param shift: ？
    :return:
    """
    if int(vector[0]) == 0 and int(vector[1]) == 0:
        pass
    else:
        pt1 = pt
        pt2 = (int(pt1[0] + vector[0]*lengthTimes),
               int(pt1[1] + vector[1]*lengthTimes))
        cv2.line(img,pt1,pt2,color,thickness,lineType,shift)
        vx = pt2[0] - pt1[0]
        vy = pt2[1] - pt1[1]
        v  = math.sqrt(vx ** 2 + vy ** 2)
        ux = vx / v
        uy = vy / v
        # 矢印の幅の部分
        w = 5
        h = 10
        ptl = (int(pt2[0] - uy*w - ux*h), int(pt2[1] + ux*w - uy*h))
        ptr = (int(pt2[0] + uy*w - ux*h), int(pt2[1] - ux*w - uy*h))
        # 矢印の先端を描画する
        cv2.line(img,pt2,ptl,color,thickness,lineType,shift)
        cv2.line(img,pt2,ptr,color,thickness,lineType,shift)

def cvVerticalArrow(img, x, vector, lengthTimes, color, isSigned=False, thickness=1, lineType=8, shift=0):
    vx, vy = vector
    if isSigned:
        verticalVector = (0, -vx)
        baseY = img.shape[0] * 1 / 3  # 画面の下から1/3の高さ
    else:
        verticalVector = (0, -math.sqrt(vx ** 2 + vy ** 2))
        baseY = img.shape[0] * 1 / 2  # 画面下端から20px上
    cvArrow(img, (x, baseY), verticalVector,
            lengthTimes, color, thickness, lineType, shift)

def cvLine(img, pt1, pt2, color, thickness=1):
    pt1 = (int(pt1[0]), int(pt1[1]))
    pt2 = (int(pt2[0]), int(pt2[1]))
    cv2.line(img, pt1, pt2, color, thickness)

# TODO: cvLineから書き直して、ベクトル描画もそれに合わせて直すべき
def cvLine2(img, pt1, pt2, color, thickness=1):
    pt1 = (int(pt1[0]), int(pt1[1]))
    pt2 = (int(pt2[0]), int(pt2[1]))
    cv2.line(img, pt1, pt2, WHITE, thickness+2)
    cv2.line(img, pt1, pt2, color, thickness)

def cvLineGraph(img, x, pitchX, vector, nextVector, lengthTimes, color, isSigned=False, thickness=1, lineType=8, shift=0):
    vx, vy = vector
    nvx, nvy = nextVector
    if isSigned:
        verticalVector = (0, -vx)
        nextVerticalVector = (0, -nvx)
        baseY = img.shape[0] * 1 / 3  # 画面の下から1/3の高さ
    else:
        verticalVector = (0, -math.sqrt(vx ** 2 + vy ** 2))
        nextVerticalVector = (0, -math.sqrt(nvx ** 2 + nvy ** 2))
        baseY = img.shape[0] * 1 / 2  # 画面下端から20px上
    cvLine(img, (x, baseY+verticalVector[1]*lengthTimes), (x+pitchX, baseY+nextVerticalVector[1]*lengthTimes),
            color, thickness)
def cvXAxis(img, isSigned, thickness=1):
    if isSigned:
        baseY = img.shape[0] * 1 / 3  # 画面の下から1/3の高さ
    else:
        baseY = img.shape[0] * 1 / 2  # 画面下端から20px上
    cvArrow(img, (0, baseY), (img.shape[1], 0), 1, BLACK, thickness)

#TODO: vector.x、vector.yで呼べるようにする（utilsの方も）

# 追跡中と検出中に呼ばれるのでメソッドにしている
def drawVelocityVectorsInStrobeMode(frameToDisplay, positionHistory,
                                    numFramesDelay, numStrobeModeSkips,
                                    velocityVectorsHistory,
                                    color=BLUE, thickness=5):
    for i in range(len(positionHistory) - numFramesDelay - 1):
        if i % numStrobeModeSkips == 0 and \
                        velocityVectorsHistory[i] is not None:
            cvArrow(
                frameToDisplay,
                positionHistory[i - numFramesDelay],
                velocityVectorsHistory[i],
                4, color, thickness
            )
            # if shouldDrawVelocityVectorsVerticallyInStrobeMode:
            #     cvVerticalArrow(
            #         frameToDisplay, spaceBetweenVerticalVectors*i,
            #         velocityVectorsHistory[i],
            #         4, color, isSigned, thickness
            #     )

def drawVelocityVectorsVerticallyInStrobeMode(frameToDisplay, positionHistory,
                                              velocityVectorsHistory, numFramesDelay,
                                              numStrobeModeSkips, spaceBetweenVerticalVectors,
                                              color=BLUE, thickness=5, isSigned=False, lengthTimes=5):
    for i in range(len(positionHistory) - numFramesDelay - 1):
        if i % numStrobeModeSkips == 0 and \
                        velocityVectorsHistory[i] is not None:
            cvVerticalArrow(
                frameToDisplay, spaceBetweenVerticalVectors*i/numStrobeModeSkips,
                velocityVectorsHistory[i],
                lengthTimes, color, isSigned, thickness
            )


# 力ベクトルを描画する
def drawForceVector(img, aclVector, positionAclBegin, gravityStrength):
    if aclVector is None:
        aclVector = (0,0)
    # 加速度ベクトル - 重力ベクトル = 力ベクトル
    vector = (aclVector[0], aclVector[1] - gravityStrength)

    if vector is not None:
        cvArrow(img, positionAclBegin, vector, 1, BLUE, 5)

def getComponentVector(vector, axis):
    if vector is None:
        return None
    elif axis is "x":
        return (vector[0], 0)  # x成分のみ使う
    elif axis is "y":
        return (0, vector[1])  # y成分のみ使う
    else:
        raise ValueError('axis is neither x nor y')

class fpsWithTick(object):
    def __init__(self):
        self._count     = 0
        self._oldCount  = 0
        self._freq      = 1000 / cv2.getTickFrequency()
        self._startTime = cv2.getTickCount()
    def get(self):
        nowTime         = cv2.getTickCount()
        diffTime        = (nowTime - self._startTime) * self._freq
        self._startTime = nowTime
        fps             = (self._count - self._oldCount) / (diffTime / 1000.0)
        self._oldCount  = self._count
        self._count     += 1
        fpsRounded      = round(fps, 1)
        return fpsRounded

def getSubtractedFrame(frameFore, frameBackground, diffBgFg, iterations):
    # まずカメラ画像をHSVに変換する
    frameNowHsv        = cv2.cvtColor(frameFore,       cv2.COLOR_BGR2HSV)
    # 保存しておいた画像もHSVに変換する
    frameBackgroundHsv = cv2.cvtColor(frameBackground, cv2.COLOR_BGR2HSV)
    # 変換した2つの画像をH,S,V各要素に分割する
    frameFgH, frameFgS, frameFgV = cv2.split(frameNowHsv)
    frameBgH, frameBgS, frameBgV = cv2.split(frameBackgroundHsv)
    # 差分計算
    diffH = cv2.absdiff(frameFgH, frameBgH)
    diffS = cv2.absdiff(frameFgS, frameBgS)
    diffV = cv2.absdiff(frameFgV, frameBgV)
    # 差分が閾値より大きければTrue
    maskH = diffBgFg < diffH
    maskS = diffBgFg * 2 < diffS  # 自動露出補正対策
    maskV = diffBgFg < diffV
    # 配列（画像）の高さ・幅
    height = frameFgH.shape[0]
    width  = frameFgH.shape[1]
    # 背景画像と同じサイズの配列生成
    im_mask_h = numpy.zeros((height, width), numpy.uint8)
    im_mask_s = numpy.zeros((height, width), numpy.uint8)
    im_mask_v = numpy.zeros((height, width), numpy.uint8)
    # Trueの部分（背景）は白塗り
    im_mask_h[maskH] = 255
    im_mask_s[maskS] = 255
    im_mask_v[maskV] = 255
    # 積集合（HSVのどれか1つでもdiffBgFgより大きい差があれば真）
    im_mask = cv2.bitwise_or(im_mask_h, im_mask_s)
    im_mask = cv2.bitwise_or(im_mask  , im_mask_v)
    # ノイズ除去
    # 8近傍
    element8 = numpy.array([[1,1,1],
                            [1,1,1],
                            [1,1,1]], numpy.uint8)
    # cv2.morphologyEx(hTarget, cv2.MORPH_CLOSE, element8, hTarget, None, iterations)
    cv2.morphologyEx(im_mask, cv2.MORPH_OPEN, element8, im_mask, None, iterations)
    return cv2.bitwise_and(frameFore, frameFore, mask=im_mask)

def getBackProjectFrame(frame, roi_hist):
    # HSV色空間に変換
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # バックプロジェクションの計算
    dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

    # 8近傍
    element8 = numpy.array([[1,1,1],
                            [1,1,1],
                            [1,1,1]], numpy.uint8)
    # オープニング
    cv2.morphologyEx(dst, cv2.MORPH_OPEN, element8, dst, None, 2)
    return dst

def getMaskByHsv(src, hueMin, hueMax, valueMin, valueMax, gamma=96, sThreshold=5,
              shouldProcessGaussianBlur=False, gaussianBlurKernelSize=5,
              shouldProcessClosing=True, iterations=1):

    _hueMin = hueMin / 2
    _hueMax = hueMax / 2

    src = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(src, numpy.array((
        _hueMin,              # H最小値
        2 ** sThreshold - 1,  # S最小値
        valueMin              # V最小値
    )), numpy.array((
        _hueMax,              # H最大値
        255,                  # S最大値
        valueMax)))           # V最大値

    # 後処理する

    if shouldProcessClosing:
        # 8近傍
        element8 = numpy.array([[1,1,1],
                                [1,1,1],
                                [1,1,1]], numpy.uint8)
        # クロージング
        # cv2.morphologyEx(hTarget, cv2.MORPH_CLOSE, element8, hTarget, None, iterations)
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, element8, mask, None, iterations)
        # anchor – アンカー点．
        # デフォルト値は(-1,-1) で、アンカーがカーネルの中心にあることを意味します

    if shouldProcessGaussianBlur:
        # ガウシアンフィルタを用いて画像の平滑化を行います．
        # GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]]) -> dst
        # ksize must pair of odd. (5,5),(7,7),(9,9)...
        size = 2 * gaussianBlurKernelSize - 1
        cv2.GaussianBlur(mask, (size,size), 0, mask)

    return mask

def scan_color(frame, x, y, w, h):
    """
    矩形内の色相Hue、明度Valueの最小値・最大値を求める
    :param frame: カメラ画像
    :param x: int 矩形左上の点のx座標
    :param y: int 矩形左上の点のy座標
    :param w: int 矩形の横幅
    :param h: int 矩形の高さ
    :return:
    """
    hueArray   = []
    valueArray = []
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    for iy in range(x, x+w):
        for ix in range(y, y+h):
            dot = frameHSV[ix][iy]
            # 色相
            if dot[0] == 0:
                pass
            else:
                dotHue = dot[0] * 2
                hueArray.append(dotHue)
                if 1000 < len(hueArray):
                    hueArray.pop()
            # 明度
            if dot[2] == 0:
                pass
            else:
                dotValue = dot[2]
                valueArray.append(dotValue)
                if 1000 < len(valueArray):
                    valueArray.pop()

    if 0 < len(hueArray):
        hueMax   = max(hueArray)
        hueMin   = min(hueArray)
    else:
        hueMin   = -1
        hueMax   = -1
    if 0 < len(valueArray):
        valueMax = max(valueArray)
        valueMin = min(valueArray)
    else:
        valueMin = -1
        valueMax = -1
    return hueMin, hueMax, valueMin, valueMax

def drawCalibrationTarget(frame, x, y, r):
    height, width, numChannels = frame.shape
    maskOfSquare = numpy.zeros((height, width), dtype=numpy.uint8)
    maskOfCircle = numpy.zeros((height, width), dtype=numpy.uint8)
    FILL = -1
    cv2.rectangle(maskOfSquare, (x-2*r,y-2*r), (x+2*r,y+2*r), 255, FILL)  # 白い正方形
    cv2.circle   (maskOfCircle, (x, y)       , r            , 255, FILL)  # 白い円
    maskOutOfCircle = 255 - maskOfCircle                                  # 円の外側が白い
    mask = 255 - cv2.bitwise_and(maskOfSquare, maskOutOfCircle)           # 黒いターゲットマーク
    frameOfRectangleWithoutCircle = numpy.zeros((height, width, 3), dtype=numpy.uint8)
    cv2.merge((mask, mask, mask), frameOfRectangleWithoutCircle)
    frame[:] = cv2.bitwise_and(frame, frameOfRectangleWithoutCircle)

def pasteRect(src, dst, frameToPaste, dstRect, interpolation = cv2.INTER_LINEAR):
    """
    入力画像の部分矩形画像をリサイズして出力画像の部分矩形に貼り付ける
    :param src:     入力画像
    :type  src:     numpy.ndarray
    :param dst:     出力画像
    :type  dst:     numpy.ndarray
    :param srcRect: (x, y, w, h)
    :type  srcRect: tuple
    :param dstRect: (x, y, w, h)
    :type  dstRect: tuple
    :param interpolation: 補完方法
    :return: None
    """

    height, width, _ = frameToPaste.shape
    # x0, y0, w0, h0 = 0, 0, width, height

    x1, y1, w1, h1 = dstRect

    # コピー元の部分矩形画像をリサイズしてコピー先の部分矩形に貼り付ける
    src[y1:y1+h1, x1:x1+w1] = \
        cv2.resize(frameToPaste[0:height, 0:width], (w1, h1), interpolation = interpolation)
    # Python: cv.Resize(src, dst, interpolation=CV_INTER_LINEAR) → None
    # Parameters:
    # src – input image.
    # dst – output image; it has the size dsize (when it is non-zero) or
    # the size computed from src.size(), fx, and fy; the type of dst is the same as of src.
    # dsize –
    # output image size; if it equals zero, it is computed as:
    # dsize = Size(round(fx*src.cols), round(fy*src.rows))
    # Either dsize or both fx and fy must be non-zero.
    # fx –
    # scale factor along the horizontal axis; when it equals 0, it is computed as
    # (double)dsize.width/src.cols
    # fy –
    # scale factor along the vertical axis; when it equals 0, it is computed as
    # (double)dsize.height/src.rows
    # interpolation –
    # interpolation method:
    # INTER_NEAREST - a nearest-neighbor interpolation
    # INTER_LINEAR - a bilinear interpolation (used by default)
    # INTER_AREA - resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire’-free results. But when the image is zoomed, it is similar to the INTER_NEAREST method.
    # INTER_CUBIC - a bicubic interpolation over 4x4 pixel neighborhood
    # INTER_LANCZOS4 - a Lanczos interpolation over 8x8 pixel neighborhood

    dst[:] = src
