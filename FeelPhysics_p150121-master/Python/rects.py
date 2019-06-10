# coding=utf-8
__author__ = 'weed'

THICKNESS = 5

import cv2

def outlineRect(image, rect, color):
    """
    画像の中に指定した色の枠線を描画する
    :param image: 画像データ
    :type  image: numpy.ndarray
    :param rect : (x, y, w, h)
    :type  rect : tuple
    :param color: BGR値 例：(255, 0, 0)
    :type  color: tuple
    :return: None
    """
    if rect is None:
        return
    x, y, w, h = rect
    cv2.rectangle(image, (x, y), (x+w, y+h), color, THICKNESS)
    # Python: cv.Rectangle(img, pt1, pt2, color, thickness=1, lineType=8, shift=0) → None
    # Parameters:
    # img – Image.
    # pt1 – Vertex of the rectangle.
    # pt2 – Vertex of the rectangle opposite to pt1 .
    # rec – Alternative specification of the drawn rectangle.
    # color – Rectangle color or brightness (grayscale image).
    # thickness – Thickness of lines that make up the rectangle. Negative values, like CV_FILLED , mean that the function has to draw a filled rectangle.
    # lineType – Type of the line. See the line() description.
    # shift – Number of fractional bits in the point coordinates.

def outlineRectWithTitle(image, rect, color, title):
    """
    左上にタイトルの付いた枠線を描画する
    :param image: 画像データ
    :type  image: numpy.ndarray
    :param rect : (x, y, w, h)
    :type  rect : tuple
    :param color: BGR値 例：(255, 0, 0)
    :type  color: tuple
    :param title: タイトル
    :type  title: str
    :return: None
    """
    if rect is None:
        return
    outlineRect(image, rect, color)

    #putText(img, text, org, fontFace, fontScale,
    # color[, thickness[, linetype[, bottomLeftOrigin]]]) -> None

    x, y, w, h = rect
    location=(x, y - (THICKNESS + 1)/2)

    fontface=cv2.FONT_HERSHEY_PLAIN
    fontscale=2.0
    cv2.putText(image, title, location,
                fontface, fontscale, color, (THICKNESS + 1)/2)

def copyRect(src, dst, srcRect, dstRect,
             interpolation = cv2.INTER_LINEAR):
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

    x0, y0, w0, h0 = srcRect
    x1, y1, w1, h1 = dstRect

    # コピー元の部分矩形画像をリサイズしてコピー先の部分矩形に貼り付ける
    dst[y1:y1+h1, x1:x1+w1] = \
        cv2.resize(src[y0:y0+w0, x0:x0+w0], (w1, h1), interpolation = interpolation)
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

def swapRects(src, dst, rects,
              interpolation = cv2.INTER_LINEAR):
    """
    2つの矩形画像を交換する
    :param src:   入力画像
    :type  src:   numpy.ndarray
    :param dst:   出力画像
    :type  dst:   numpy.ndarray
    :param rects: 矩形のリスト
    :type  rects: list[tuple]
    :param interpolation: 補完方法
    :return: None
    """

    # 出力画像に入力画像をコピー
    if dst is not src:
        dst[:] = src

    # 矩形の数が2以上でなければ終了する
    numRects = len(rects)
    if numRects < 2:
        return

    # 最後の矩形の画像を一時的に保存する
    x, y, w, h = rects[numRects - 1]
    temp = src[y:y+h, x:x+w].copy()

    # 矩形2から矩形3に
    i = numRects - 2
    while i >= 0:
        copyRect(src, dst, rects[i], rects[i+1], interpolation)
        i -= 1

    copyRect(temp, dst, (0, 0, w, h), rects[0], interpolation)

    # len(rects)が2のとき
    # rect0とrect1とを交換（swap）したい
    # しかし・・・
    # copyRect(rect0, rect1)「rect0をrect1にコピー」
    # copyRect(rect1, rect0)「rect1をrect0にコピー」
    # ・・・だと、すでにrect1にはrect0がコピーされているので
    # rect0をrect0にコピーすることになってしまう。
    # そこで、rect0をrect1にコピーする前にrect1をtempに退避させる
    # temp = src(rect1).copy
    # copyRect(rect0, rect1)
    # copyRect(temp , rect0)