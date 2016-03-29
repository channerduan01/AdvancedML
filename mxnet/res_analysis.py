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

def setComparison(A,B,data,show=False):
    print "share ratio for A:%f (%d)" %(float(len(A&B))/float(len(A)),len(A))
    print "share ratio for B:%f (%d)" %(float(len(A&B))/float(len(B)),len(B))
    
    print "from A->B,obtained:%d,lost:%d" %(len(A-B),len(B-A))
    if not show:
        return
    list_ = []
    for i in list(A-B):
        list_.append("+  %d" %i)
        list_.append(data[i,0])
    drawFigures(list_)
    list_ = []
    for i in list(B-A):
        list_.append("-  %d" %i)
        list_.append(data[i,0])
    drawFigures(list_)

def drawSet(set_,label):
    list_ = []
    for i in list(set_):
        list_.append("%d" %label[i])
        list_.append(data[i,0])
    drawFigures(list_)
    
if not 'test_data' in dir():
    test_data = genfromtxt(open('test_ready.csv','r'), delimiter=',', dtype='f8')
    test_data = test_data.reshape((len(test_data),1,28,28))
def coupledPredict(list_vesions,epoch=100):
    with Timer() as t:
        res_ = np.zeros((len(test_data),10),np.float)
        for v_ in list_versions:
            print "%s_predicting" %v_
            model = mx.model.FeedForward.load(
                prefix=v_,
                epoch=epoch
                )
            res = model.predict(test_data)
            res_ += res
        print "coupled_predicting"
        res = np.argmax(res_,1)
        savetxt('submission.csv', np.c_[np.arange(1, res.size+1), res], delimiter=',', header='ImageId,Label', fmt='%d', comments='')
    print "=> prediction cost: %s s" % t.secs


plt.gray()
#list_versions = ['cnn_01_','cnn_07_','cnn_08_','cnn_09_','cnn_10_']  # 0.99014
#list_versions = ['cnn_01_','cnn_05_','cnn_06_','cnn_08_','cnn_09_']  # 0.99086
#list_versiosns = ['cnn_01_','cnn_01_1','cnn_01_2','cnn_01_3']  # 0.98957
#list_versions = ['cnn_01_','cnn_05_','cnn_06_','cnn_08_','cnn_09_','cnn_10_','cnn_10_1']  # 0.99014
#list_versions = ['cnn_01_','cnn_05_','cnn_06_','cnn_08_','cnn_09_','cnn_10_']   # 0.99014
#list_versions = ['cnn_01_1','cnn_05_','cnn_06_','cnn_08_','cnn_09_']    # 0.99114(100) 0.99086(80)
list_versions = ['cnn_01_2','cnn_05_','cnn_06_','cnn_08_','cnn_09_'] 

epoch_num = 100

#coupledPredict(list_versions,epoch_num)

data = valid_data
label = valid_label.astype(np.uint8)
list_ = []
res_ = np.zeros((len(data),10),np.float)
for v_ in list_versions:
    print "%s:" %v_
    model = mx.model.FeedForward.load(
        prefix=v_,
        epoch=epoch_num
        )
    indices, res = validationAnalysis(model,data,label)
    res_ += res
    list_.append(indices)
    
res = np.argsort(res_,1)[:,-1]
coupled_predict = np.where(res != label)[0]
print "coupled correct-rate:%6f (err_num:%d)" %(1.0-float(len(coupled_predict))/float(len(label)),len(coupled_predict))


#setComparison(set(list_[0]),set(list_[-3]),data,True)

#validationAnalysis(model,data,label,show=True)

## draw the shared error images
#size = len(list_)
#corr = np.zeros((size,size),dtype=np.int)
#base_set = set()
#for i in range(size):
#    for j in range(size):
#            corr[i,j] = len(set(list_[i])&set(list_[j]))
#    if len(base_set) == 0:
#        base_set = set(list_[i])
#    else:
#        base_set = base_set & set(list_[i])
#drawSet(base_set,label)
















