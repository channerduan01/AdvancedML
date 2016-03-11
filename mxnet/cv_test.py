# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 13:32:26 2016

@author: channerduan
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def drawFigures(params):
    length = len(params)
    if (length < 2 or length%2 == 1):
        raise Exception("illegal input")
    for i in range(0,length,4):
        plt.figure()
        plt.subplot(121)
        plt.title(params[i])
        plt.axis('off')
        plt.imshow(params[i+1])
        if (i+2 < length):
            plt.subplot(122)
            plt.title(params[i+2])
            plt.axis('off')
            plt.imshow(params[i+3])
    return

random_state = np.random.RandomState(None)
def elastic_distortion(image, alpha, sigma):
    s = image.shape
    dx = gaussian_filter(random_state.rand(*s) * 2 - 1, sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter(random_state.rand(*s) * 2 - 1, sigma, mode="constant", cval=0) * alpha
    x, y = np.meshgrid(np.arange(s[0]), np.arange(s[1]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
    return map_coordinates(image, indices, order=1).reshape(s)

plt.gray()
image = cv2.cvtColor(cv2.imread('model_image.jpg'),cv2.COLOR_RGB2GRAY)

list_ = []
for i in range(20):
    list_.append(i)
    list_.append(elastic_distortion(image,500,30))
drawFigures(list_)