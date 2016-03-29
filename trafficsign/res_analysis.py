# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 13:32:26 2016

@author: channerduan
"""

import sys
sys.path.append("../common")
from timer import Timer
from numpy import genfromtxt, savetxt

import matplotlib.pyplot as plt
import numpy as np
import mxnet as mx

def drawFigures(params,width=6):
    length = len(params)
    if (length < 2 or length%2 == 1 or width < 1):
        raise Exception("illegal input")
    for i in range(0,length,2):
        if (i%(width*2) == 0):
            plt.figure()
        plt.subplot(100+width*10+(i%(width*2))/2+1)
        plt.title(params[i])
        plt.axis('off')
        plt.imshow(params[i+1])
    return

def validationAnalysis(model,data,label,show=False):
    raw_predict = model.predict(data)
    res = np.argsort(raw_predict,1)
    p_label = res[:,-1].astype(np.uint8)
    indices = label != p_label
    res_err = res[indices]
    print "correct-rate:%6f (err_num:%d)" %(1.0-float(len(res_err))/float(len(res)),len(res_err))
    if not show:
        return np.where(indices)[0], raw_predict
    correction = label[indices]
    data_source = data[indices]
    list_ = []
    for i in range(len(correction)):
        list_.append('%d,%d,%d  (%d)' %(res_err[i,-1],res_err[i,-2],res_err[i,-3],correction[i]))
        list_.append(data_source[i,0])
    drawFigures(list_)

plt.gray()
prefix = 'cnn_01_'
epochs_num = 60

model = mx.model.FeedForward.load(
    prefix=prefix,
    epoch=epochs_num
)
#indices, res = validationAnalysis(model,train_data,train_label)
indices, res = validationAnalysis(model,test_data,test_label,True)















