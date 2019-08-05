# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 11:34:18 2018

@author: seraj
"""

 
import sys
#sys.path.append('C:\Users\seraj\Desktop\myapppython\YOLO\darkflow-master')
from darkflow.net.build import TFNet
import cv2
import matplotlib.pyplot as plt
#%config.Inlinebackend.figure_format = 'svg' 
#Could not create cuDNN handle when convnets are 
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
options = {
        'model':'cfg/yolov2-voc.cfg',
        'load':'bin/yolov2-voc.weights',
        'threshold':0.3,
        'gpu': 1.0
        }

tfnet = TFNet(options)  #print modlw arch

img = cv2.imread('sample_dog.jpg')
img.shape

result = tfnet.return_predict(img)
print(result)

for x in result:
        t1 = (x['topleft']['x'], x['topleft']['y'])  # top left
        b1 = (x['bottomright']['x'], x['bottomright']['y'])  # bottomright
        labl = x['label']
        img = cv2.rectangle(img, t1, b1,(0.,255,0),2)
        img = cv2.putText(img,labl, t1,cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
print("--------------")
print(img.shape)
plt.imshow(img)
plt.show()











