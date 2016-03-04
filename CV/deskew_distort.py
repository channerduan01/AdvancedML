# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 23:18:33 2016

@author: yx
"""

import math
import cv
import cv2
from PIL import Image
import numpy as np
from scipy import signal
from scipy.ndimage import filters
import matplotlib.pyplot as plt
from numpy.random import random_integers
from scipy.signal import convolve2d
from numpy import genfromtxt, savetxt

print 'load data...'
dataset = genfromtxt(open('train.csv','r'), delimiter=',', dtype='f8')[1:]
target = [x[0] for x in dataset]
train = [x[1:] for x in dataset]
test = genfromtxt(open('test.csv','r'), delimiter=',', dtype='f8')[1:]
test = list(test)

SZ = 20 #need to be changed with the image's size
affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
def deskew(img):
     m = cv2.moments(img)
     if abs(m['mu02']) < 1e-2:
         return img.copy()
     skew = m['mu11']/m['mu02']
     M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
     img = cv2.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
     #print img.shape
     #img = img.reshape(784,)
     n,m = img.shape
     #img = img.reshape(n*m,)
     return img  

def norm_width(im):
    im = np.uint8(im)

    out=cv2.resize(im,(20,20),interpolation=cv2.INTER_CUBIC)
    return out

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
    
def el_distor(im,  kernel_dim, Sigma, alpha):
    
    out = np.zeros(im.shape)
    
    displace_x = np.array([[random_integers(-1, 1) for x in xrange(im.shape[0])] \
                            for y in xrange(im.shape[1])]) * alpha
    displace_y = np.array([[random_integers(-1, 1) for x in xrange(im.shape[0])] \
                            for y in xrange(im.shape[1])]) * alpha
                   
    kernel = create_gaussian_kernel(kernel_dim, Sigma)
    #kernel = cv2.getGaussianKernel(kernel_dim,Sigma)
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
N = len(test)
for n in range(M):
    train[n] = train[n].reshape(math.sqrt(784),math.sqrt(784))
    train[n] = norm_width(train[n])
    
    train[n] = deskew(train[n])
    
    train[n] = el_distor(train[n],5,1.5,4)
    
 #   train[n] = train[n].reshape(math.sqrt(400),math.sqrt(400))
 #   plt.gray()
 #   plt.figure()
 #   plt.imshow(train[n])

for n in range(N):
    test[n] = test[n].reshape(math.sqrt(784),math.sqrt(784))
    test[n] = norm_width(test[n])
    
    test[n] = deskew(test[n])
    test[n] = el_distor(test[n],5,1.5,4)

    

np.savetxt("some_data/distor_train.csv", train, delimiter=",", fmt='%s')
np.savetxt("some_data/distor_test.csv", test, delimiter=",", fmt='%s') 

     