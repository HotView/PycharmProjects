import cv2
import numpy
import utils
"""这里定制了一些基本的卷积滤波器，甚至比OpenCV里一些现成的滤波器还要基本。"""

"""文件中增加两个类，卷积滤波类和锐化滤波类"""

def strokeEdges(src,dst,blurKsize = 7,edgeKsize = 5):
    if blurKsize>=3:
        blurredSrc = cv2.medianBlur(src,blurKsize)
        graySrc = cv2.cvtColor(blurredSrc,cv2.COLOR_BGR2GRAY)
    else:
        graySrc  =cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    cv2.Laplacian(graySrc,cv2.CV_8U,graySrc,ksize=edgeKsize)
    normalizedInverseAlpha =  (1.0/255)*(255-graySrc)
    channels  =cv2.split(src)

    for channel in channels:
        channel[:] = channel*normalizedInverseAlpha
    cv2.merge(channels,dst)
class VConvolutionFilter(object):
    def __init__(self,kernel):
        self._kernel = kernel

    def apply(self,src,dst):
        cv2.filter2D(src,-1,self._kernel,dst)

class SharpFilter(VConvolutionFilter):
    def __init__(self):
        kernel = numpy.array([[-1,-1,-1],
                              [-1,9,-1],
                              [-1,-1,-1]])
        VConvolutionFilter.__init__(self,kernel)

class FindEdgesFilter(VConvolutionFilter):
    def __init__(self):
        kernel = numpy.array([[-1,-1,-1],
                              [-1,8,-1],
                              [-1,-1,-1]])
        VConvolutionFilter.__init__(self,kernel)

class BlurFilter(VConvolutionFilter):
    def __init__(self):
        kernel = numpy.array([[0.04,0.04,0.04,0.04,0.04] for i in range(5)])
        VConvolutionFilter.__init__(self,kernel)

"""同时具有模糊和锐化的效果"""
class EmbossFilter(VConvolutionFilter):
    def __init__(self):
        kernel = numpy.array([[-2,-1,0],
                              [-1,1,1],
                              [0,1,2]])
        VConvolutionFilter.__init__(self,kernel)

