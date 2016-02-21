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
from numpy import savetxt
import logging

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
    
def fit(network,shape,iteration,baseModel=None):
    num_epochs = iteration
    batch = 128

    head = '%(asctime)-15s Node %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    
    print 'start load data...'
    train = mx.io.CSVIter(
        data_csv='data.csv',
#        data_csv='anne_data_train.csv',     
        
        data_shape=shape,
        label_csv='data_label.csv',
        label_shape=(1,),
        batch_size=batch)
    valid = mx.io.CSVIter(
        data_csv='valid.csv',
#        data_csv='anne_data_valid.csv',
        
        data_shape=shape,
        label_csv='valid_label.csv',
        label_shape=(1,),
        batch_size=batch)
    print 'start trainning...'
    model_args = {}
    if baseModel is not None:
        model_args = {'arg_params' : baseModel.arg_params,
                  'aux_params' : baseModel.aux_params,
                  'begin_epoch' : baseModel.begin_epoch}        
        
    model = mx.model.FeedForward(
        ctx                = mx.cpu(),
        symbol             = network,
        num_epoch          = num_epochs,
        learning_rate      = 0.002,
        momentum           = 0.9,
        wd                 = 0.00001,
        initializer        = mx.init.Xavier(factor_type="in", magnitude=2.34),
        **model_args)
    model.fit(
        X                  = train,
        eval_data          = valid,
        kvstore            = None,
        batch_end_callback = mx.callback.Speedometer(True,20),
        epoch_end_callback = None)
    print 'finish trainning'
    return model,valid


def output(model,shape):
    test = mx.io.CSVIter(
        data_csv='test_ready.csv',
#        data_csv='anne_test.csv',
        
        data_shape=shape,
        batch_size=500)
    with Timer() as t:
        res = np.argmax(model.predict(test),1)
    print "=> prediction spent: %s s" % t.secs
    savetxt('submission.csv', np.c_[np.arange(1, res.size+1), res], delimiter=',', header='ImageId,Label', fmt='%d', comments='')
    return

prefix = 'mnist'
iteration = 12

#shape = (128,)          # for Anne...
##shape = (784,)          # for MLP
shape = (1, 28, 28)     # for convolution

model_loaded = None
#model_loaded = mx.model.FeedForward.load(prefix, 20)
with Timer() as t:
#    model,valid = fit(get_mlp(),shape,iteration,model_loaded)
    model,valid = fit(get_lenet(),shape,iteration,model_loaded)
    model.save(prefix, iteration)
print "=> training spent: %s s" % t.secs

#valid.reset()
#res = np.argmax(model.predict(valid),1)

#model_loaded = mx.model.FeedForward.load(prefix, iteration)
#output(model_loaded,shape)





