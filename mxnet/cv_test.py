# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 13:32:26 2016

@author: channerduan
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def drawFigures(params,width=4):
    length = len(params)
    if (length < 2 or length%2 == 1 or width < 1):
        raise Exception("illegal input")
    for i in range(0,length,2):
        if (i%(width*2) == 0):
            plt.figure()
        plt.subplot(100+width*10+(i%(width*2))/2+1)
        plt.title(params[i])
        plt.axis('off')
        plt.imshow(params[i+1])
    return

random_state = np.random.RandomState(None)
def elastic_distortion(image, alpha, sigma):
    s = image.shape
    dx = gaussian_filter(random_state.rand(*s) * 2 - 1, sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter(random_state.rand(*s) * 2 - 1, sigma, mode="constant", cval=0) * alpha
    x, y = np.meshgrid(np.arange(s[0]), np.arange(s[1]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
    return map_coordinates(image, indices, order=1).reshape(s)

def deskew(image):
     n,m = image.shape
     affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
     m = cv2.moments(image)
     if abs(m['mu02']) < 1e-2:
         return image.copy()
     skew = m['mu11']/m['mu02']
     M = np.float32([[1, skew, -0.5*n*skew], [0, 1, 0]])
     return cv2.warpAffine(image,M,(n, n),flags=affine_flags)
      
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

def shear_image(img):
    rows,cols = img.shape
    img = np.uint8(img)
    a = np.random.random_integers(0,4)
    a = 4
    flag = np.random.random_integers(0,1)
    flag = 0
    third_p = [1,20]
    if flag == 1 :
        pts1 = np.float32([[1+a,1],[10+a,1],third_p])
        pts2 = np.float32([[15-a,1],[25-a,1],third_p])
    elif flag == 0 :
        pts1 = np.float32([[15-a,1],[25-a,1],third_p])
        pts2 = np.float32([[1+a,1],[10+a,1],third_p])
    M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst

plt.gray()
image = cv2.cvtColor(cv2.imread('model_image.jpg'),cv2.COLOR_RGB2GRAY)

#list_ = []
#for i in range(20):
#    list_.append(i)
#    list_.append(elastic_distortion(image,300,15))
#drawFigures(list_)


list_ = []
for i in range(20):
    list_.append('original image:%d' %i)
    list_.append(train_data[i,0])
    list_.append('distort')
    
#    list_.append(shear_image(train_data[i,0]))
#    list_.append(elastic_distortion(train_data[i,0],300,10))
    list_.append(move_to_centre(elastic_distortion(train_data[i,0],75,6),train_data[i,0]))
#    list_.append(deskew(train_data[i,0]))    
    
drawFigures(list_,width=6)