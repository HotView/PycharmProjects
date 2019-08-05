# -*- coding: utf-8 -*-
import os
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data

PATH = os.getcwd()
logdir = 'D:/tmp/tensorflow/mnist/logdir'  # you will need to change this!!!
embed_count = 1600
mnist = input_data.read_data_sets(PATH + "/data/",one_hot=True)
x = tf.placeholder(tf.float32,[None,784])
W =tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x,W)+b#预测值

y_ = tf.placeholder(tf.float32,[None,10])
cross_entroy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entroy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# Train
for _ in range(1000):
    batch_xs,batch_y = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_y})
## 评估模型
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))

