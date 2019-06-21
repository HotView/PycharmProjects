#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Liu
import cv2
import numpy as np
img = cv2.imread("laser-v.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
np.savetxt("laser-v.txt",gray[100:200,:],fmt="%.3d")
