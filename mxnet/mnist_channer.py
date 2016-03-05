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

batch_size = 256
basic_lr = 0.002
anneal_lr = 0.98 #0.993
tParam = None
def batch_callback(param):
    global tParam
    tParam = param
    if (param.nbatch % 30 == 0):
        print 'nbatch:%d' % (param.nbatch)

def epoch_callback(epoch, symbol, arg_params, aux_params):
    global train_data
    global train_label
    print 'first 10 train mark: %f' %np.sum(train_data[0:10])
#    if sgd_opt.lr > 0.00003:
#        sgd_opt.lr = anneal_lr**epoch*basic_lr # 0.993
#        print 'Epoch[%d] learning rate:%f' % (epoch, sgd_opt.lr)
    index = np.arange(len(train_data))
    np.random.shuffle(index)
    train_data = train_data[index]
    train_label = train_label[index]

def createModel():
    sgd_opt = opt.SGD(
    learning_rate=anneal_lr**0*basic_lr,
        momentum=0.9, 
        wd=0.0001, 
        rescale_grad=(1.0/batch_size))
    model = mx.model.FeedForward(
        ctx                = mx.cpu(),
        symbol             = get_dnn(),
        num_epoch          = None,
        numpy_batch_size   = batch_size,
        optimizer          = sgd_opt,
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

prefix = 'cnn_00_' # 28*28,, shuffle, anneal learning rate
load_target = 0
iteration_target = 20
##shape = (784,)          # for MLP
data_shape = (1, 28, 28)     # for convolution

#--------------------- data preparation
if not 'train_data' in dir() or not 'train_label' in dir():
    train_data = genfromtxt(open('data.csv','r'), delimiter=',', dtype='f8')
    train_data = train_data.reshape((len(train_data),1,28,28))
    train_label = genfromtxt(open('data_label.csv','r'), delimiter=',', dtype='f8')
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
    model = mx.model.FeedForward.load(prefix, load_target)
    model.num_epoch = iteration_target+load_target
else:
    model = createModel()
    model.num_epoch = iteration_target
#--------------------- model training
with Timer() as t:
    model.fit(
    X = train_data,
    y = train_label,
    eval_data= valid,
    kvstore = None,
    batch_end_callback = batch_callback,
    epoch_end_callback = epoch_callback)
print "=> trained[%d->%d] cost: %s s" %(load_target,load_target+iteration_target,t.secs)
model.save(prefix, iteration)
#--------------------- prediction
# makePrediction(model,date_shape)









