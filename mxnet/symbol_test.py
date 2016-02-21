# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 13:32:26 2016

@author: channerduan
"""

import mxnet as mx
import logging
import numpy as np

def demoSymbol():
    A = mx.sym.Variable('A')
    B = mx.sym.Variable('B')
    C = A*B
    D = mx.sym.Variable('D')
    E = C+D
    G = mx.sym.Group([C,E])
    a = mx.nd.ones((1,2))*1.5
    b = mx.nd.ones((1,2))
    d = mx.nd.ones((1,2))
    executor = G.bind(ctx=mx.cpu(),args={'A':a,'B':b,'D':d})
    executor.forward()
    print executor.outputs[0].asnumpy()
    print executor.outputs[1].asnumpy()
    return

def get_mlp():
    """
    multi-layer perceptron
    """
    data = mx.symbol.Variable('data')
    fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=128)
    act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
    fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
    act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
    fc3  = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=10)
    mlp  = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')
    return mlp

#demoSymbol()
#print get_mlp()

def fit(network, (train, val)):
    num_epochs = 1

    head = '%(asctime)-15s Node %(message)s'
    logging.logMultiprocessing
    logging.basicConfig(level=logging.DEBUG, format=head)

    model = mx.model.FeedForward(
        ctx                = mx.cpu(),
        symbol             = network,
        num_epoch          = num_epochs,
        learning_rate      = 0.1,
        momentum           = 0.9,
        wd                 = 0.00001,
        initializer        = mx.init.Xavier(factor_type="in", magnitude=2.34))

    model.fit(
        X                  = train,
        eval_data          = val,
        kvstore            = None,
        batch_end_callback = mx.callback.Speedometer(True,40),
        epoch_end_callback = None)


def get_iterator():
    data_shape = (784, )
    batch_size = 100
    data_dir = 'mnist/'
    
    train           = mx.io.MNISTIter(
        image       = data_dir + "train-images-idx3-ubyte",
        label       = data_dir + "train-labels-idx1-ubyte",
        input_shape = data_shape,
        batch_size  = batch_size,
        shuffle     = True,
        flat        = True,
        num_parts   = 1,
        part_index  = 0)

    val = mx.io.MNISTIter(
        image       = data_dir + "t10k-images-idx3-ubyte",
        label       = data_dir + "t10k-labels-idx1-ubyte",
        input_shape = data_shape,
        batch_size  = batch_size,
        flat        = True,
        num_parts   = 1,
        part_index  = 0)
        
    
#    print train.getdata().asnumpy().shape
#    print train.getindex()    
#    print train.getlabel().asnumpy().shape
    
#    print train.getlabel().asnumpy()
#    i = 0
#    while train.iter_next():
#        train.next()
#        print train.iter_next()
#        i += 1
#        print i
#        
#    print train.iter_next().getlabel().asnumpy()
#    print help(train)
    
    return (train, val)

(train, val) = get_iterator()
#A = train.getdata().asnumpy()
#B = train.getlabel().asnumpy()
#C = train.getindex()

A = train.data
B = train.label
C = train.index

#fit(get_mlp(), get_iterator())


    
