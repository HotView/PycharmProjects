import cv2
import os
import numpy as np
import tensorflow as tf
import pickle
def unpickle(filename):
    with open(filename,'rb') as f:
        d = pickle.load(f,encoding='bytes')
        return d
def rebuild(filepath):
    data1 = unpickle("cifar-10-batches-py/data_batch_1")
    print(data1.keys())
    data_2d = data1[b'data']
    for image in data_2d:
        print(image.shape)
        dim = (224,224)
        image = np.reshape(image,(3,32,32)).T
        resized = cv2.resize(image,dim)

filepath = "E:/cat_and_dog/cifar"
rebuild(filepath)