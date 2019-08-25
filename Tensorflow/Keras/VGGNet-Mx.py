import tensorflow as tf
from scipy.misc import imresize,imread
from keras.utils import np_utils
import numpy as np
from mxnet import nd
import mxnet
import d2lzh as d2l
class vgg16():
    def __init__(self,imgs,n_classes):
        self.imgs = imgs
        self.parameters = []
        self.n_classes = n_classes
        self.convlayers()
        self.fc_layers()
        self.probs = self.fc8
    def saver(self):
        return tf.train.Saver()
    def maxpool(self,name,input_data):
        out = tf.nn.max_pool(input_data,[1,2,2,1],[1,2,2,1],padding="SAME",name=name)
        return out
    def conv(self,name,input_data,out_channels):
        in_channel = input_data.get_shape()[-1]
        with tf.variable_scope(name):
            kernel = tf.get_variable("weights",[3,3,in_channel,out_channels],dtype=tf.float32,trainable=False)
            biases = tf.get_variable("biases",[out_channels],dtype=tf.float32,trainable=False)
            conv_res = tf.nn.conv2d(input_data,kernel,[1,1,1,1],padding="SAME")
            res = tf.nn.bias_add(conv_res,biases)
            out = tf.nn.relu(res,name=name)
        self.parameters+=[kernel,biases]
        return out
    def fc(self,name,input_data,out_channel,trainables = True):
        shape = input_data.get_shape().as_list()
        if len(shape)==4:
            size = shape[-1]*shape[-2]*shape[-3]
        else:
            size = shape[1]
        input_data_flat = tf.reshape(input_data,[-1,size])
        with tf.variable_scope(name):
            weights = tf.get_variable(name="weights",shape=[size,out_channel],dtype=tf.float32,trainable=trainables)
            biases = tf.get_variable(name="biases",shape=[out_channel],dtype=tf.float32,trainable=trainables)
            res = tf.matmul(input_data_flat,weights)
            out = tf.nn.relu(tf.nn.bias_add(res,biases))
        self.parameters+=[weights,biases]
        return out
    def convlayers(self):
        #conv1
        self.conv1_1 = self.conv("conv1_1",self.imgs,64)
        print(self.conv1_1.shape,"#########")
        self.conv1_2 = self.conv("conv1_2",self.conv1_1,64)
        self.pool1 = self.maxpool("pool1",self.conv1_2)

        # conv2
        self.conv2_1 = self.conv("conv2_1",self.pool1,128)
        self.conv2_2 = self.conv("conv2_2", self.conv2_1, 128)
        self.pool2 = self.maxpool("pool2",self.conv2_2)

        # conv3
        self.conv3_1 = self.conv("conv3_1",self.pool2,256)
        self.conv3_2 = self.conv("conv3_2",self.conv3_1,256)
        self.conv3_3 = self.conv("conv3_3",self.conv3_2,256)
        self.pool3 = self.maxpool("pool3",self.conv3_3,)

        # conv4
        self.conv4_1 = self.conv("conv4_1",self.pool3,512)
        self.conv4_2 = self.conv("conv4_2",self.conv4_1,512)
        self.conv4_3 = self.conv('conv4_3',self.conv4_2,512)
        self.pool4 = self.maxpool("pool4",self.conv4_3)

        # conv5
        self.conv5_1 = self.conv("conv5_1",self.pool4,512)
        self.conv5_2 = self.conv("conv5_2",self.conv5_1,512)
        self.conv5_3 = self.conv('conv5_3',self.conv5_2,512)
        self.pool5 = self.maxpool("pool5",self.conv5_3)
    def fc_layers(self):
        self.fc6 = self.fc("fc6",self.pool5,out_channel=4096,trainables=False)
        self.fc7 = self.fc("fc7",self.fc6,4096,trainables=False)
        self.fc8 = self.fc("fc8",self.fc7,self.n_classes)
    def load_weights(self,weight_file,sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        print(len(keys),"len key")
        print(len(self.parameters))
        for i ,k in enumerate(keys):
            if i not in [30,31]:
                sess.run(self.parameters[i].assign(weights[k]))
        print('-------------all done------------------')
x_imgs = tf.placeholder(tf.float32,[None,96,96,1])
vgg=  vgg16(x_imgs,10)
y_imgs = tf.placeholder(tf.int32,[None,10])
fc3_cat_and_dog = vgg.probs
batch_size =500
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc3_cat_and_dog, labels=y_imgs))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
train_iter, test_iter = d2l.load_data_fashion_mnist(200, resize=96)
epoch = 5
for i in range(epoch):
    j = 0
    for features, labels in train_iter:
        X_train = nd.transpose(features,axes=(0,2,3,1)).asnumpy()
        y_train = labels.asnumpy()
        Y_train = np_utils.to_categorical(y_train,10)
        sess.run(optimizer,feed_dict={x_imgs:X_train,y_imgs:Y_train})
        j = j+1
        if j%10==0:
            print(sess.run(loss,feed_dict={x_imgs:X_train,y_imgs:Y_train}))