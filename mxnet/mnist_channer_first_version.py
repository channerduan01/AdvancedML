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

tParam = None
    
def fit(network,shape,num_epochs,baseModel=None):
    batch = 256
    learning_rate = 0.001

#    head = '%(asctime)-15s Node %(message)s'
#    logging.basicConfig(level=logging.DEBUG, format=head)
    
#    print 'start load data...'
#    train = mx.io.CSVIter(
#        data_csv='data.csv',
#        data_shape=shape,
#        label_csv='data_label.csv',
#        label_shape=(1,),
##        shuffle=True,
#        batch_size=batch)
    valid = mx.io.CSVIter(
        data_csv='valid.csv',
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

    sgd_opt = opt.SGD(learning_rate=learning_rate, momentum=0.9, wd=0.0001, rescale_grad=(1.0/batch))
    model = mx.model.FeedForward(
        ctx                = mx.cpu(),
        symbol             = network,
        num_epoch          = num_epochs,
        optimizer          = sgd_opt,
        initializer        = mx.init.Uniform(0.05),
        **model_args)
    def batch_callback(param):
        global tParam
        tParam = param
        if (param.nbatch % 20 == 0):
            print 'nbatch:%d' % (param.nbatch)
    def epoch_callback(epoch, symbol, arg_params, aux_params):
        global train_data
        global train_label
        print 'first 10 train mark: %f' %np.sum(train_data[0:10])
        if sgd_opt.lr > 0.00003:
#            sgd_opt.lr *= 0.993
            sgd_opt.lr *= 0.99
#        print 'nepoch:%d, learning rate:%f' % (epoch, sgd_opt.lr)
        np.random.shuffle(shuffle_index)
        train_data = train_data[shuffle_index]
        train_label = train_label[shuffle_index]
    model.fit(
        X = train_data,
        y = train_label,
#        eval_data = (valid_data,valid_label),
#        X                  = train,
        eval_data          = valid,
        kvstore            = None,
        batch_end_callback = batch_callback,
        epoch_end_callback = epoch_callback)
    print 'finish trainning'
    return model,valid

def output(model,shape):
    test = mx.io.CSVIter(
        data_csv='test_ready.csv',
        data_shape=shape,
        batch_size=500)
    with Timer() as t:
        res = np.argmax(model.predict(test),1)
    print "=> prediction spent: %s s" % t.secs
    savetxt('submission.csv', np.c_[np.arange(1, res.size+1), res], delimiter=',', header='ImageId,Label', fmt='%d', comments='')
    return



#train_data = genfromtxt(open('data.csv','r'), delimiter=',', dtype='f8')
#train_data = train_data.reshape((len(train_data),1,28,28))
#train_label = genfromtxt(open('data_label.csv','r'), delimiter=',', dtype='f8')
#valid_data = train_data.copy()
#valid_label = train_label.copy()

shuffle_index = np.arange(len(train_data))

prefix = 'mnist'
iteration = 30

##shape = (784,)          # for MLP
shape = (1, 28, 28)     # for convolution

model_loaded = None
#model_loaded = mx.model.FeedForward.load(prefix, 210)
with Timer() as t:
#    model,valid = fit(get_mlp(),shape,iteration,model_loaded)
    model,valid = fit(get_dnn(),shape,iteration,model_loaded)
    model.save(prefix, iteration)
print "=> training spent: %s s" % t.secs

#valid.reset()
#res = np.argmax(model.predict(valid),1)

#model_loaded = mx.model.FeedForward.load(prefix, iteration)
#output(model_loaded,shape)





