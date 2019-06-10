# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 20:54:17 2018

@author: seraj
"""
import cv2
from darkflow.net.build import TFNet
import numpy as np
import time

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

capture = cv2.VideoCapture(0)
colors = [tuple(255 * np.random.rand(3)) for i in range(5)]
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out11 = cv2.VideoWriter('output.avi',fourcc, 20.0, (1280,720))

while (capture.isOpened()):
    stime = time.time()
    ret, frame = capture.read()
    if ret:
        results = tfnet.return_predict(frame)
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            frame = cv2.rectangle(frame, tl, br, color, 7)
            frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow('frame', frame)
        out11.write(frame)
        print('FPS {:.1f}'.format(1 / (time.time() - stime)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        capture.release()
        cv2.destroyAllWindows()
        break
print("finished")
out11.release()