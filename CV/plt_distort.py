# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 13:55:34 2016

@author: yx
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt, savetxt

def drawFigures(params):
    length = len(params)
    if (length < 2 or length%2 == 1):
        raise Exception("illegal input")
    for i in range(0,length,4):
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


def distort(train):
    img = train.copy()
    A = img.shape[0] / 5.0
    w = 0.7/ img.shape[1]
    shift = lambda x: A * np.sin(2.0*np.pi*x * w)
    for i in range(img.shape[0]):
        img[:,i] = np.roll(img[:,i], int(shift(i)))
    return img
   
    
dataset = genfromtxt(open('some_data/20pixel_sample.csv','r'), delimiter=',', dtype='f8')[0:]

data_shape = (20,20)

train = dataset.reshape((len(dataset),data_shape[0],data_shape[1]))

res = np.zeros_like(train)
    
M = len(train)

for i in range(len(train)):
    res[i] = distort(train[i])
    plt.figure()
    drawFigures(['',train[i],'',res[i]])   


