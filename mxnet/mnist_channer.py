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



def get_mlp():
    """
    multi-layer perceptron
    """
    data = mx.symbol.Variable('data')
    fc0  = mx.symbol.FullyConnected(data = data, name='fc0', num_hidden=128)
    act0 = mx.symbol.Activation(data = fc0, name='relu0', act_type="relu")  

    fc1  = mx.symbol.FullyConnected(data = act0, name='fc1', num_hidden=128)
    act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
    fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 128)
    act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
    fc3  = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=10)
    mlp  = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')
    return mlp

def get_lenet():
    """
    LeCun, Yann, Leon Bottou, Yoshua Bengio, and Patrick
    Haffner. "Gradient-based learning applied to document recognition."
    Proceedings of the IEEE (1998)
    """
    data = mx.symbol.Variable('data')
    # first conv
    conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=32)
    tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
    pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    # second conv
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=64)
    tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
    pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    # first fullc
    flatten = mx.symbol.Flatten(data=pool2)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
    tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")
    # second fullc
    fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=10)
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
    fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=10)
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
    print 'data refresh start...'
    for i in range(len(train_data)):
        train_data_tmp[i,0] = elastic_distortion(train_data[i,0],500,30)
    print 'first 100 train mark: %.0f - %.0f' %(np.sum(train_data[0:100]),np.sum(train_data_tmp[0:100]))
    print 'data refresh finish...'

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

def makePrediction(model,shape):
    test = mx.io.CSVIter(
        data_csv='test_ready.csv',
        data_shape=shape,
        batch_size=512)
    with Timer() as t:
        res = np.argmax(model.predict(test),1)
    print "=> prediction cost: %s s" % t.secs
    savetxt('submission.csv', np.c_[np.arange(1, res.size+1), res], delimiter=',', header='ImageId,Label', fmt='%d', comments='')
    return
    
#prefix = 'cnn_00_' # 28*28,, shuffle,anneal_lr=0.994,basic_lr=0.003,batch_size=256,, 98.5+
#prefix = 'cnn_01_' # 28*28,, shuffle,anneal_lr=0.994,basic_lr=0.003,batch_size=128,, 98.7+
#prefix = 'cnn_02_' # 28*28,, shuffle,anneal_lr=0.991,basic_lr=0.0018,batch_size=128,, 98.6+
#prefix = 'cnn_03_' # 28*28,, shuffle,anneal_lr=0.994,basic_lr=0.003,batch_size=64,, 98.6+
#prefix = 'cnn_04_' # 28*28,, shuffle,anneal_lr=0.99,basic_lr=0.002,batch_size=50,, 98.7+
prefix = 'cnn_05_' # 28*28,, shuffle,distortion(500,30),anneal_lr=0.994,basic_lr=0.003,batch_size=128,,

check_point_step = 20
load_target = 0
iteration_target = 100
##shape = (784,)          # for MLP
data_shape = (1, 28, 28)     # for convolution

#--------------------- data preparation
if not 'train_data' in dir() or not 'train_label' in dir():
    train_data = genfromtxt(open('data.csv','r'), delimiter=',', dtype='f8')
    train_data = train_data.reshape((len(train_data),1,28,28))
    train_label = genfromtxt(open('data_label.csv','r'), delimiter=',', dtype='f8')
train_data_tmp = np.zeros_like(train_data)
if not 'valid_data' in dir():
    valid_data = mx.io.CSVIter(
        data_csv='valid.csv',
        data_shape=data_shape,
        label_csv='valid_label.csv',
        label_shape=(1,),
        batch_size=512)
valid_data.reset()
#--------------------- model establish
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
        eval_data= valid_data,
        kvstore = None,
        batch_end_callback = batch_callback,
        epoch_end_callback = epoch_callback
        )
    model.save(prefix, model.num_epoch)
print "=> trained[%d->%d] cost: %s s" %(load_target,load_target+iteration_target,t.secs)
#--------------------- prediction
#makePrediction(model,data_shape)


#--------------------- draw test
#plt.gray()
#list_ = []
#for i in range(8):
#    list_.append('original image:%d' %i)
#    list_.append(train_data[i,0])
#    list_.append('distort')
#    list_.append(elastic_distortion(train_data[i,0],500,30))
#drawFigures(list_)





