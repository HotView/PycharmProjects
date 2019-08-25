from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.optimizers import SGD,Adam,RMSprop
import matplotlib.pyplot as plt
import d2lzh as d2l
from mxnet import nd
#60000/60000 [==============================] - 38s 633us/step - loss: 0.0797 - acc: 0.9697
IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32

BATCH_SIZE = 128
NB_EPOCH = 20
NB_CLASSES = 10
VERBOSE = 1
VALLIDATION_SPILIT = 0.2
OPTIM = Adam()
(X_train,y_train),(X_test,y_test) = cifar10.load_data()
print('X_tarin shape:',X_train.shape)
print('train samples:',X_train.shape[0])
print('test shape:',X_test.shape[0])
Y_train = np_utils.to_categorical(y_train,NB_CLASSES)
Y_test = np_utils.to_categorical(y_test,NB_CLASSES)
# float and categorical
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /=255
X_test /=255
# network
model = Sequential()
model.add(Conv2D(32,(3,3),padding='same',input_shape=(96,96,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',optimizer=OPTIM,metrics=['accuracy'])
train_iter, test_iter = d2l.load_data_fashion_mnist(60000, resize=96)
for X_test,y_test in test_iter:
    pass
X_test = nd.transpose(X_test,axes=(0,2,3,1)).asnumpy()
y_test = y_test.asnumpy()
y_test = np_utils.to_categorical(y_test, 10)
print("######")
print(X_test.shape)
print(y_test.shape)
print("######")
for X_train, y_train in train_iter:
    pass
X_train = nd.transpose(X_train, axes=(0, 2, 3, 1)).asnumpy()
y_train = y_train.asnumpy()
Y_train = np_utils.to_categorical(y_train, 10)
for i in range(20):
    model.fit(X_train,Y_train,batch_size=128,)
    score = model.evaluate(X_test,y_test)
    print("test score:", score[0],"test accuracy:", score[1])
        #score = model.evaluate(X_test,y_test)
#score = model.evaluate(X_test,Y_test,batch_size = BATCH_SIZE,verbose=VERBOSE)
#print("test score:",score[0])
#print("test accuracy:",score[1])



