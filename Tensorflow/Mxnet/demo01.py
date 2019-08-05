import mxnet as mx
import d2lzh as d2l
from mxnet import nd
from mxnet.gluon import nn

ctx = d2l.try_gpu()
print(ctx)
x = nd.array([1, 2, 3],ctx =mx.gpu(0))
#B = nd.random.uniform(shape=(2, 3), ctx=mx.gpu(1))
print(x)
#print(B)