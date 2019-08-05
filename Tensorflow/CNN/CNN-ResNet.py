import keras.backend as K
from  keras.models import Model
from keras.layers import Input,Dense,Conv2D,DepthwiseConv2D,SeparableConv2D
from keras.layers import Flatten,MaxPool2D,AvgPool2D,GlobalAvgPool2D,UpSampling2D
from keras.layers import BatchNormalization,concatenate,add,Dropout,ReLU,Lambda,Activation,LeakyReLU
import os
from time import time
import numpy as np
def resnet(input_shape,n_classes):

    def conv_bn_r1(x,f,k=1,s=1,p = "same"):
        x =Conv2D(f,k,strides=s,padding=p)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x
    def identity_block(tensor,f):
        x =conv_bn_r1(tensor,f)
        x = conv_bn_r1(x,f,3)
        x = Conv2D(4*f,1)(x)
        x = BatchNormalization()(x)

        x = add([x,tensor])
        output = ReLU()(x)
        return output
    def conv_block(tensor,f,s):
        x = conv_bn_r1(tensor,f)
        x = conv_bn_r1(x,f,3,s)
        x = Conv2D(4*f,1)(x)
        x =BatchNormalization()(x)

        shortcut = Conv2D(4*f,1,strides=s)(tensor)
        shortcut =BatchNormalization()(shortcut)

        x = add([x,shortcut])
        output = ReLU()(x)
        return output
    def resnet_block(x,f,r,s=2):
        x = conv_block(x,f,s)
        for _ in range(r-1):
            x = identity_block(x,f)
        return x
    input = Input(input_shape)

    x = conv_bn_r1(input,64,7,2)
    x = MaxPool2D(3,strides=2,padding='same')(x)

    x = resnet_block(x,64,3,1)
    x = resnet_block(x,128,4)
    x = resnet_block(x,256,6)
    x = resnet_block(x,512,3)

    x = GlobalAvgPool2D()(x)

    output =Dense(n_classes,activation='softmax')(x)

    model = Model(input,output)
    return model
IN_PUT_SHAPE = (96,96,1)
N_CLASSES = 10
K.clear_session()
model = resnet(IN_PUT_SHAPE,N_CLASSES)
model.summary()
