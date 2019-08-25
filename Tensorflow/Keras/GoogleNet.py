import keras.backend as K
from  keras.models import Model
from keras.layers import Input,Dense,Conv2D,DepthwiseConv2D,SeparableConv2D
from keras.layers import Flatten,MaxPool2D,AvgPool2D,GlobalAvgPool2D,UpSampling2D
from keras.layers import BatchNormalization,concatenate,add,Dropout,ReLU,Lambda,Activation,LeakyReLU
import os
from time import time
import numpy as np

def googlenet(input_data,n_classes):
    def inception_block(x,f):
        t1 = Conv2D(f[0],1,activation='relu')(x)

        t2 = Conv2D(f[1],1,activation='relu')(x)
        t2 = Conv2D(f[2],3,padding="same",activation='relu')(t2)

        t3 = Conv2D(f[3],1,activation='relu')(x)
        t3 = Conv2D(f[4],5,padding='same',activation='relu')(t3)

        t4 = MaxPool2D(3,1,padding='same')(x)
        t4 = Conv2D(f[5],1,activation='relu')(t4)

        output = concatenate([t1,t2,t3,t4])
        return output
    input = Input(input_data)

    x= Conv2D(64,7,strides=2,padding="same",activation='relu')(input)
    x= MaxPool2D(3,strides=2,padding='same')(x)

    x= Conv2D(64,1,activation='relu')(x)
    x = Conv2D(192,3,padding='same',activation='relu')(x)
    x = MaxPool2D(3,strides=2)(x)

    x= inception_block(x,[64,96,128,16,32,32])
    x= inception_block(x,[128,128,192,32,96,64])
    x = MaxPool2D(strides=2,padding='same')(x)

    x = inception_block(x,[192,96,208,16,48,64])
    x = inception_block(x, [160, 112, 224, 24, 64, 64])
    x = inception_block(x, [128, 128, 256, 24, 64, 64])
    x = inception_block(x, [112, 144, 288, 32, 64, 64])
    x = inception_block(x, [256, 160, 320, 32, 128, 128])
    x = MaxPool2D(3,strides=2,padding='same')(x)

    x = inception_block(x,[256,160,320,32,128,128])
    x =inception_block(x,[384,192,384,48,128,128])
    print(x.get_shape())

    x =AvgPool2D(3,strides=1)(x)
    x = Dropout(0.4)(x)

    x = Flatten()(x)
    output = Dense(n_classes,activation='softmax')(x)
    model = Model(input,output)
    return model
IN_PUT_SHAPE = (96,96,1)
N_CLASSES = 10
K.clear_session()
model = googlenet(IN_PUT_SHAPE,N_CLASSES)
model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy")

model.summary()