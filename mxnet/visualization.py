# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 20:36:02 2016

@author: channerduan
"""

import mxnet as mx
import numpy as np
import math
import matplotlib.pyplot as plt

def draw(img,title='',size=7):
    plt.figure(figsize=(size, size))
    plt.axis('off')
    plt.title(title)
    plt.imshow(img)
    return
    
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

def normalize(matrix):
    min_ = np.min(matrix)
    max_ = np.max(matrix)
    return (matrix-min_)/(max_-min_)
    
def getConvWeight(name,side):
    param_dict = model.arg_params
    weights = param_dict.get(name).asnumpy()
    s = weights.shape
    weights = weights.reshape(s[0]*s[1],side,side)
    return normalize(weights)    

def showOverall(name,side,size=5):
    weights = getConvWeight(name,side)
    num = len(weights)
    length = np.ceil(math.sqrt(num)).astype(np.int)
    canvas = np.zeros((length*side+length,length*side+length),np.float)
    x = 0
    for i in range(num):
        if i > 0 and i % length == 0:
            x += 1
        y = i-x*length
#        print "(%d,%d)" %(x,y)
        canvas[x*side+x:(x+1)*side+x,y*side+y:(y+1)*side+y] = weights[i]
    draw(canvas,name,5)
    return

def showDetails(name,side):
    weights = getConvWeight(name,side)
    list_ = []
    for i in range(len(weights)):
        list_.append('filter:%d' %i)
        list_.append(weights[i])
    drawFigures(list_)

model = mx.model.FeedForward.load(
    prefix='cnn_00_',
    epoch=100
    )
showOverall('convolution2_weight',4)
showOverall('convolution3_weight',5,12)

model = mx.model.FeedForward.load(
    prefix='cnn_01_',
    epoch=100
    )
showOverall('convolution0_weight',4)
showOverall('convolution1_weight',5,12)


#showDetails('convolution0_weight',4)
#showDetails('convolution1_weight',5)













