import d2lzh as d2l
import numpy as np
import cv2
batch_size = 128
print("start")
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
i = 0
for X,y in train_iter:
    if i<1:
        dataX= X.asnumpy()
        n,c,w,h = dataX.shape
        outX = dataX.reshape(n,w,h,c)
        datay = y.asnumpy()
        # for i in outX:
        #     cv2.imshow("gray",i)
        #     cv2.waitKey(500)
        i = i+1
    else:
        dataX = X.asnumpy()
        n, c, w, h = dataX.shape
        dataX = dataX.reshape(n, w, h, c)
        outX = np.concatenate([outX,dataX])
        datay = np.concatenate([datay,y.asnumpy()])
        print(outX.shape)
        print(datay.shape)
# np.save("fashion_mnist_96_X",outX)
# np.save("fashion_mnist_y",datay)
