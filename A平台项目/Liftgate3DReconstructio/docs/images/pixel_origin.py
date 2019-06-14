#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: eph

import os

import numpy as np
import matplotlib.pyplot as plt
import cv2


image = np.zeros((12, 16))
cv2.circle(image, (8, 6), 5, 1)

fig = plt.figure(figsize=(5, 4), dpi=100)
ax = fig.add_subplot(1, 1, 1)

ax.pcolor(image[::-1], cmap=plt.get_cmap('Greys'), antialiaseds=True)
ax.add_artist(plt.Circle((8.5, 5.5), 5, fc='None', ec='0.6'))

ax.set_xticks(np.arange(image.shape[1] + 1))
ax.set_yticks(np.arange(image.shape[0] + 1))
ax.set_xticklabels(np.arange(image.shape[1] + 1))
ax.set_yticklabels(image.shape[0] - np.arange(image.shape[0] + 1))
ax.tick_params(axis='x', which='major', colors='r',
               top=1, bottom=0, labeltop=1, labelbottom=0)
ax.tick_params(axis='y', which='major', colors='r',
               left=1, right=0, labelleft=1, labelright=0)
ax.grid(color='0.6')
ax.text(-1, image.shape[0] + 2, 'Image Corner Origin',
        fontsize=16, ha='left', va='center', color='r')
ax.add_artist(plt.Circle((0, image.shape[0]), 0.15,
                         fc='r', ec='None', clip_on=False, zorder=3))
ax.add_artist(plt.Circle((0.5, image.shape[0] - 0.5), 0.15,
                         fc='b', ec='None', clip_on=False, zorder=3))

ax.set_xticks(np.arange(image.shape[1]) + 0.5, minor=True)
ax.set_yticks(np.arange(image.shape[0]) + 0.5, minor=True)
ax.set_xticklabels(np.arange(image.shape[1]), minor=True)
ax.set_yticklabels(image.shape[0] - 1 - np.arange(image.shape[0]), minor=True)
ax.tick_params(axis='x', which='minor', colors='b',
               top=0, bottom=1, labeltop=0, labelbottom=1)
ax.tick_params(axis='y', which='minor', colors='b',
               left=0, right=1, labelleft=0, labelright=1)
ax.tick_params('both', length=0, pad=5, which='both')
ax.text(image.shape[1] + 1, -2, 'Pixel Center Origin',
        fontsize=16, ha='right', va='center', color='b')

ax.add_artist(plt.Rectangle((4, 5), 9, 1, fc='w', ec='0.6', zorder=2))
ax.text(8.5, 5.5, 'cv2.circle(image, (8, 6), 5, ...',
        ha='center', va='center')

plt.savefig(os.path.splitext(__file__)[0] + '.png', dpi=100,
            format='png', bbox_inches='tight')
