# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 14:53:28 2016

@author: yx
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import random_integers
from scipy.signal import convolve2d
from numpy import genfromtxt, savetxt

#only distort


print 'load data...'
dataset = genfromtxt(open('some_data/20pixel_train.csv','r'), delimiter=',', dtype='f8')[0:]

train = [x[0:] for x in dataset]


def create_gaussian_kernel(dim, sigma):

    if dim % 2 == 0:
        raise ValueError("Kernel dimension should be odd")

    kernel = np.zeros((dim, dim), dtype=np.float16)

    center = dim/2

    variance = sigma ** 2
    
    coef = 1. / (2 *math.pi* variance)
    
    den = 2*variance

    for x in range(0, dim):
        for y in range(0, dim):
            x_val = abs(x - center)
            y_val = abs(y - center)
            num = -1*(x_val**2 + y_val**2)
            
            
            kernel[x,y] = coef * np.exp(num/den)
    
    return kernel/sum(sum(kernel))
    
def distor(im,  kernel_dim, Sigma, alpha):
    
    out = np.zeros(im.shape)
    
    displace_x = np.array([[random_integers(-1, 1) for x in xrange(im.shape[0])] \
                            for y in xrange(im.shape[1])]) * alpha
    displace_y = np.array([[random_integers(-1, 1) for x in xrange(im.shape[0])] \
                            for y in xrange(im.shape[1])]) * alpha
                   
    kernel = create_gaussian_kernel(kernel_dim, Sigma)
    displace_x = convolve2d(displace_x, kernel)
    displace_y = convolve2d(displace_y, kernel)
    
    for row in xrange(im.shape[1]):
        for col in xrange(im.shape[0]):
            low_x = row + int(math.floor(displace_x[row, col]))
            high_x = row + int(math.ceil(displace_x[row, col]))

            low_y = col + int(math.floor(displace_y[row, col]))
            high_y = col + int(math.ceil(displace_y[row, col]))

            if  high_x >= im.shape[1] -1 or high_y >= im.shape[0] - 1:
                continue

            res = im[low_x, low_y]/4 + im[low_x, high_y]/4 + \
                    im[high_x, low_y]/4 + im[high_x, high_y]/4

            out[row, col] = res   
    n,m = out.shape
    out = out.reshape(n*m,)    
    return out
M = len(train)

for n in range(M):

    train[n] = train[n].reshape(math.sqrt(400),math.sqrt(400))

    train[n] = distor(train[n],17,3,8)
  
#    train[n] = train[n].reshape(math.sqrt(400),math.sqrt(400))
#    plt.gray()
#    plt.figure()
#    plt.imshow(train[n])



np.savetxt("some_data/distort_train.csv", train, delimiter=",", fmt='%s')
