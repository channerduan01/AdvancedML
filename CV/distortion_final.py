# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 11:54:43 2016

@author: yx
"""

#updated distortion

import math
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt, savetxt

import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


def elastic_distortion(image, alpha, sigma, random_state):

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    return map_coordinates(image, indices, order=1).reshape(shape)

def norm_img(img):
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            img[x,y] = np.floor(img[x,y]-img.min()*255.0/(img.max()-img.min()))
    return img


def deskew(img):
     n,m = img.shap
     SZ = n
     affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
     m = cv2.moments(img)
     if abs(m['mu02']) < 1e-2:
         return img.copy()
     skew = m['mu11']/m['mu02']
     M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
     img = cv2.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
     n,m = img.shape
     #img = img.reshape(n*m,)
     return img  

def change_width(img,size):
    im = np.uint8(img)
    out=cv2.resize(im,(size,size),interpolation=cv2.INTER_CUBIC)
    n,m = out.shape
#    out = out.reshape(n*m,)   

    return out



def drawFigures(params):
    length = len(params)
    if (length < 2 or length%2 == 1):
        raise Exception("illegal input")
    for i in range(0,length,4):
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


def test_A():
    img = cv2.imread('a.jpg')
    im = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imo = im.copy()

    s = []
    plt.gray()
    for i in range(9):

        im = elastic_distortion(imo,550,19,None)
        s.append(im)
        #im = norm_img(im)
    plt.subplot(221)
    plt.axis('off')
    plt.imshow(s[0])
    plt.subplot(222)
    plt.axis('off')
    plt.imshow(s[1])
    plt.subplot(223)
    plt.axis('off')
    plt.imshow(s[2])
    plt.subplot(224)
    plt.axis('off')
    plt.imshow(s[3])   


def test_sample_number():
    dataset = genfromtxt(open('some_data/29pixel_sample.csv','r'), delimiter=',', dtype='f8')[0:]

    data_shape = (29,29)
    train = dataset.reshape((len(dataset),data_shape[0],data_shape[1]))
    res = np.zeros_like(train)
    
    for i in range(len(train)):

   
         res[i] = elastic_distortion(train[i],85,5.1,None)
         res[i] = cv2.convertScaleAbs(res[i])
         plt.gray()
         plt.figure()
   
         drawFigures(['',train[i],'',res[i]]) 


def distort_all_train() :   
    dataset = genfromtxt(open('train.csv','r'), delimiter=',', dtype='f8')[1:]
    train = [x[1:] for x in dataset]
    for i in range(len(train)):

        train[i] = train[i].reshape(math.sqrt(784),math.sqrt(784))
        train[i] = elastic_distortion(train[i],85,5,1,None)
        #train[i] = norm_img(train[i])
        train[i] = train[i].reshape(28*28,)

    np.savetxt("some_data/Andrew_distort_train.csv", train, delimiter=",", fmt='%s')
    
test_A()
plt.figure()
test_sample_number()