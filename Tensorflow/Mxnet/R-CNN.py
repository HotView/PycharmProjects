import d2lzh as d2l
from mxnet import contrib,gluon,image,nd
import numpy as np
np.set_printoptions(2)
img = image.imread("C:/Users/Q/Pictures/Saved Pictures/7.jpg").asnumpy()
h,w = img.shape[0:2]
print(h,w)
X = nd.random.uniform(shape=(1,3,h,w))
Y = contrib.nd.MultiBoxPrior(X,sizes= [0.75,0.5,0.25],ratios = [1,2,0.5])
Y.shape
def show_bboxes(axes,bboxes,labels = None,colors = None):
    def _make_list(obj,default_values = None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj,(list,tuple)):
            obj = [obj]
        return obj
    labels = _make_list(labels)
    colors = _make_list(colors,['b','g','r','m','c'])
    for i,bbox in enumerate(bboxes):
        color = colors[i%len(colors)]
        rect = d2l.bbox_to_rect(bbox.asnumpy(),color)
        axes.add_path(rect)
        if labels and len(labels)>i:
            text_color = 'k' if color =='w'else 'w'
