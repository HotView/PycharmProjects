import d2lzh as d2l
from d2lzh import resnet18
from mxnet import autograd,gluon,init
from mxnet.gluon import data as gdata,loss as gloss ,nn
import os
import sys
import cv2
import numpy as np
import shutil
import time
data_dir = "E:/Download"
train_dir = "traindata"
label_file = "trainLabels.csv"
input_dir = "input"
valid_ratio = 0.1
batch_size = 200
# def read_label_file(data_dir,label_file,train_dir,valid_ratio):
#     with open(os.path.join(data_dir,label_file),'r') as f:
#         lines = f.readlines()[1:]
#         print(lines)
#         tokens= [l.rstrip().split(',') for l in lines]
#         idx_label = dict(((int(idx),label) for idx,label in tokens))
#     labels = set(idx_label.values())
#     n_train_valid = len(os.listdir(os.path.join(data_dir,train_dir)))
#     n_train = int(n_train_valid*(1-valid_ratio))
#     assert  0<n_train<n_train_valid
#     return n_train//len(labels),idx_label
# n_train_per_label,idx_label = read_label_file(data_dir,label_file,train_dir,valid_ratio)
# print(n_train_per_label)
# print(idx_label)
labels = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
def load_data_cifar(batch_size, resize,root=os.path.join('~', '.mxnet', 'datasets', 'cifar')):
    transformer = []
    transformer += [gdata.vision.transforms.Resize(resize)]
    transformer += [gdata.vision.transforms.ToTensor()]
    transformer_train = gdata.vision.transforms.Compose(transformer)
    valid_ds = gdata.vision.ImageFolderDataset(root, flag=1)
    train_iter = gdata.DataLoader(valid_ds.transform_first(transformer_train),batch_size, shuffle=True,last_batch='keep')
    return train_iter
data_path = os.path.join(data_dir, input_dir, 'train')
train_iter =  load_data_cifar(batch_size,96,root=data_path)
i = 0
for X,y in train_iter:
    if i<1:
        dataX= X.asnumpy()
        outX = np.transpose(dataX, (0, 2, 3, 1))
        outy = y.asnumpy()
        # for i in outX:
        #     cv2.imshow("gray",i)
        #     cv2.waitKey(500)
        i = i+1
    else:
        dataX = X.asnumpy()
        dataX = np.transpose(dataX, (0, 2, 3, 1))
        outX = np.concatenate([outX,dataX])
        outy = np.concatenate([outy,y.asnumpy()])
        print(outX.shape)
        print(outy.shape)
# print(len(outy))
# for i in range(100):
#     img = outX[i]
#     print(img.shape)
#
#     print(labels[outy[i]])
#     #cv2.putText(img, labels[outy[i]], (50,100), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 255), 2,cv2.LINE_AA)
#     print(labels[outy[i]])
#     cv2.imshow("se", img)
#     cv2.waitKey(2000)
# np.save("E:/Download/kagg_cifar_train_963_X",outX)
# np.save("E:/Download/kagg_cifar_train_y",outy)