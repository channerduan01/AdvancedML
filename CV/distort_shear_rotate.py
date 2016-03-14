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
       # plt.axis('off')
        plt.imshow(params[i+1])
        if (i+2 < length):
            plt.subplot(122)
            plt.title(params[i+2])
        #    plt.axis('off')
            plt.imshow(params[i+3])
    return


def test_distort_A():
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


def add_bound(img):
    im = np.uint8(img)
    (cnts, _) = cv2.findContours(im.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)        
    for c in cnts:
        if cv2.contourArea(c) < 20:
            continue 
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(im, (x, y), (x + w, y + h), (100, 255, 0), 1)
    return im,-1*(x+w/2),-1*(y+h/2)

def move_to_centre(res,train):
    res,res_w,res_h = add_bound(res)
    train, train_w,train_h = add_bound(train)
    diff_w = train_w-res_w
    diff_h = train_h-res_h 
    
    rows,cols = res.shape

    M = np.float32([[1,0,-1*diff_w],[0,1,-1*diff_h]])
    dst = cv2.warpAffine(res,M,(cols,rows))

    return dst

def rotate_image(img):
    rows,cols = img.shape
    ang = np.random.random_integers(-60,60)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),ang,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst
    
def shear_image(img):
    rows,cols = img.shape
    img = np.uint8(img)
    a = np.random.random_integers(0,4)
    flag = np.random.random_integers(0,1)

    if flag == 1 :

        pts1 = np.float32([[1+a,1],[10+a,1],[1,17]])
        pts2 = np.float32([[15-a,1],[25-a,1],[1,17]])
    elif flag == 0 :
        pts1 = np.float32([[15-a,1],[25-a,1],[1,17]])
        pts2 = np.float32([[1+a,1],[10+a,1],[1,17]])
    M = cv2.getAffineTransform(pts1,pts2)

    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst

def test_rotate():
    dataset = genfromtxt(open('some_data/29pixel_sample.csv','r'), delimiter=',', dtype='f8')[0:]

    data_shape = (29,29)
    train = dataset.reshape((len(dataset),data_shape[0],data_shape[1]))
    res = np.zeros_like(train)
    
    for i in range(len(train)):

        
        res[i] = rotate_image(train[i])
        res[i] = cv2.convertScaleAbs(res[i])
       
        plt.gray()
        plt.figure()
   
        drawFigures(['',train[i],'',res[i]]) 


def test_shear():
    dataset = genfromtxt(open('some_data/29pixel_sample.csv','r'), delimiter=',', dtype='f8')[0:]

    data_shape = (29,29)
    train = dataset.reshape((len(dataset),data_shape[0],data_shape[1]))
    res = np.zeros_like(train)
    
    for i in range(len(train)):

   
        res[i] = shear_image(train[i])
        res[i] = cv2.convertScaleAbs(res[i])

        train[i], train_w,train_h = add_bound(train[i])

        res[i] = move_to_centre(res[i],train[i])
            
        plt.gray()
        plt.figure()
   
        drawFigures(['',train[i],'',res[i]]) 

       

def test_distort_number():
    dataset = genfromtxt(open('some_data/29pixel_sample.csv','r'), delimiter=',', dtype='f8')[0:]

    data_shape = (29,29)
    train = dataset.reshape((len(dataset),data_shape[0],data_shape[1]))
    res = np.zeros_like(train)

    
    for i in range(len(train)):

   
         res[i] = elastic_distortion(train[i],200,8,None)
         res[i] = cv2.convertScaleAbs(res[i])

         train[i], train_w,train_h = add_bound(train[i])
    
         res[i] = move_to_centre(res[i],train[i])
            
         plt.gray()
         plt.figure()
   
         drawFigures(['',train[i],'',res[i]]) 


def distort_all_train() :   
    dataset = genfromtxt(open('train.csv','r'), delimiter=',', dtype='f8')[1:]
    train = [x[1:] for x in dataset]
    for i in range(len(train)):

        train[i] = train[i].reshape(math.sqrt(784),math.sqrt(784))
        train[i] = elastic_distortion(train[i],81,5.3,None)
        train[i] = train[i].reshape(28*28,)

    np.savetxt("some_data/Andrew_distort_train3.csv", train, delimiter=",", fmt='%s')
    
#test_distort_A()
#plt.figure()
#test_distort_number()
#distort_all_train()
test_shear()
#test_rotate()