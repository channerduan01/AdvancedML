# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 14:47:22 2016

@author: channerduan
"""
import sys
sys.path.append("../common")
import mxnet as mx
from timer import Timer
import numpy as np
import cv2
import mxnet.optimizer as opt
from numpy import genfromtxt, savetxt

import matplotlib.pyplot as plt
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
def add_bound(img):
    im = np.uint8(img)
    (cnts, _) = cv2.findContours(im.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)        
    for c in cnts:
        if cv2.contourArea(c) < 20:
            continue 
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(im, (x, y), (x + w, y + h), (100, 255, 0), 1)
    return -1*(x+w/2),-1*(y+h/2)#,im
    
def move_to_centre(res,origin):
    res_w,res_h = add_bound(res)
    train_w,train_h = add_bound(origin)
#    res_w,res_h,res = add_bound(res)
#    train_w,train_h,origin = add_bound(origin)    
    rows,cols = res.shape
    M = np.float32([[1,0,-1*(train_w-res_w)],[0,1,-1*(train_h-res_h)]])
    dst = cv2.warpAffine(res,M,(cols,rows))
    return dst
def deskew(img):
     n,m = img.shape
     SZ = n
     affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
     m = cv2.moments(img)
     if abs(m['mu02']) < 1e-2:
         return img.copy()
     skew = m['mu11']/m['mu02']
     M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
     img = cv2.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
     n,m = img.shape
     return img

# 100 + 100*150 = 15100
def get_super_dnn():
    data = mx.symbol.Variable('data')
    # first conv
    conv1 = mx.symbol.Convolution(data=data, kernel=(7,7), num_filter=100)
    tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
    pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    # second conv
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(4,4), num_filter=150)
    tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
    pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    # first fullc
    flatten = mx.symbol.Flatten(data=pool2)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=300)
    tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")
    # second fullc
    fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=43)
    # loss
    lenet = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
    return lenet
    
def get_dnn():
    data = mx.symbol.Variable('data')
    # first conv
    conv1 = mx.symbol.Convolution(data=data, kernel=(4,4), num_filter=20)
    tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
    pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    # second conv
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=40)
    tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
    pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max",
                              kernel=(2,2), stride=(3,3))
    # first fullc
    flatten = mx.symbol.Flatten(data=pool2)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=150)
    tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")
    # second fullc
    fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=43)
    # loss
    lenet = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
    return lenet

def train_data_refresh():
    global train_data
    global train_label
    global train_data_tmp
    index = np.arange(len(train_data))
    np.random.shuffle(index)
    train_data = train_data[index]
    train_label = train_label[index]
    
    #skip refreshing, just for test
    train_data_tmp = train_data
    
#    print 'data refresh start...'
#    for i in range(len(train_data)):
#        train_data_tmp[i,0] = move_to_centre(elastic_distortion(train_data[i,0],75,6),train_data[i,0])
#    print 'first 100 train mark: %.0f - %.0f' %(np.sum(train_data[0:100]),np.sum(train_data_tmp[0:100]))
#    print 'data refresh finish...'

batch_size = 128
basic_lr = 0.003
anneal_lr = 0.994 #0.993
def batch_callback(param):
    if (param.nbatch % 30 == 0):
        print 'nbatch:%d' % (param.nbatch)

def epoch_callback(epoch, symbol, arg_params, aux_params):
    global model
    if (epoch > 0 and epoch % check_point_step == 0):
        model.save(prefix, epoch)
    train_data_refresh()
    opt = model.optimizer
    if opt.lr > 0.00003:
        opt.lr = anneal_lr**epoch*basic_lr
        print 'Epoch[%d] learning rate:%f' %(epoch, opt.lr)
    
def createOptimizer(epoch_process):
    sgd_opt = opt.SGD(
        learning_rate=anneal_lr**epoch_process*basic_lr,
        momentum=0.9,
        wd=0.0001,
        rescale_grad=(1.0/batch_size)
        )
    return sgd_opt
    
def createModel():
    model = mx.model.FeedForward(
        ctx                = mx.cpu(),
        symbol             = get_dnn(),
        num_epoch          = None,
        numpy_batch_size   = batch_size,
        optimizer          = createOptimizer(0),
        initializer        = mx.init.Uniform(0.05))
    return model


#prefix = 'cnn_00_' # 32*32,,direct_binary shuffle,anneal_lr=0.994,basic_lr=0.003,batch_size=128,,
prefix = 'cnn_01_' # 32*32,,enhanced_binary shuffle,anneal_lr=0.994,basic_lr=0.003,batch_size=128,,


check_point_step = 10
load_target = 0
iteration_target = 100

data_shape = (channel, edge_len, edge_len)
train_data_tmp = np.zeros_like(train_data)

if load_target > 0:
    model = mx.model.FeedForward.load(
        prefix=prefix,
        epoch=load_target
        )
    model.optimizer = createOptimizer(load_target)
    model.num_epoch = load_target+iteration_target
    model.numpy_batch_size = batch_size
else:
    model = createModel()
    model.num_epoch = iteration_target
#--------------------- model training
with Timer() as t:
    train_data_refresh()
    model.fit(
        X = train_data_tmp,
        y = train_label,
#        eval_data= valid_data,
        eval_data= (valid_data,valid_label),
        kvstore = None,
        batch_end_callback = batch_callback,
        epoch_end_callback = epoch_callback
        )
    model.save(prefix, model.num_epoch)
print "=> trained[%d->%d] cost: %s s" %(load_target,load_target+iteration_target,t.secs)

