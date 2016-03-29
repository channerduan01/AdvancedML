# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 13:32:26 2016

@author: channerduan
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
import csv

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
    
def readOneImage(path,row):
    img = cv2.imread(path)[int(row[3]):int(row[5])+1,int(row[4]):int(row[6])+1]
    return img

def readTrafficSignsTrain(rootpath):
    images = [] # images
    labels = [] # corresponding labels
    index = 0
    class_map = np.ones((43,2500),np.int)*-1
    # loop over all 43 classes
    for c in range(0,43):
        i = 0
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        gtReader.next() # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            # extract the signs,, row[3]-row[5],row[4]-row[6]
            # the 1th column is the filename
            images.append(readOneImage(prefix + row[0],row))
            labels.append(row[7]) # the 8th column is the label
            class_map[c,i] = index
            index = index+1
            i = i+1
        gtFile.close()
    return images, labels, class_map
    
def readTrafficSignsTest(rootpath):    
    images = [] # images
    labels = [] # corresponding labels
    index = 0
    indices = np.zeros((43),dtype=np.int)
    class_map = np.ones((43,2500),np.int)*-1    
    gtFile = open(rootpath + '/GT-final_test.csv')
    gtReader = csv.reader(gtFile, delimiter=';')
    gtReader.next()
    for row in gtReader:
        class_ = int(row[7])
        images.append(readOneImage(rootpath + '/' + row[0],row))
        labels.append(class_)
        class_map[class_,indices[class_]] = index
        indices[class_] = indices[class_] + 1
        index = index + 1
    return images, labels, class_map

# no matter how many channels that image contains
def change_width(img,size):
    return cv2.resize(img,(size,size),interpolation=cv2.INTER_CUBIC)

def image_pre(img,size):
    return imageEnhance(change_width(img,size),True)

 # test draw   
def showDataSamples(data,images):
    list_ = []
    for i in range(10):
        list_.append('image:%d' %i)
        list_.append(data[i,0])
    drawFigures(list_)
    list_ = []
    for i in range(10):
        list_.append('image:%d' %i)
        list_.append(images[i])
    drawFigures(list_)
   
def showDataTypes():
    list_ = []
    for class_ in range(select_range):
        img = trainImages[trainClassmap[class_,10]]
        list_.append('class:%d' %class_)
#        list_.append(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY))
#        list_.append('enhanced')
        list_.append(imageEnhance(img,True))      
#        list_.append('binary')
#        list_.append(image_pre(img,edge_len))
#        list_.append('enhanced-binary')
#        list_.append(image_pre(imageEnhance(img),edge_len))
    drawFigures(list_)

def dataTransform(data,lables,classmap):
    length = np.sum(np.sum(classmap>=0,1)[0:select_range])
    print "num of data should be: %d" %length

    data_ = np.zeros((length,channel,edge_len,edge_len),dtype='uint8')
    label_ = np.zeros((length),dtype='uint8')
    c_index = 0
    for class_ in range(select_range):
        for s_index in classmap[class_]:
            if s_index < 0:
                break
            data_[c_index] = image_pre(data[s_index],edge_len)
            label_[c_index] = lables[s_index]
            c_index = c_index+1
    print "num of result: %d" %c_index    
    return data_,label_

def dataSimpleTransform(data,labels):
    length = len(data)
    data_ = np.zeros((length,channel,edge_len,edge_len),dtype='uint8')
    label_ = np.zeros((length),dtype='uint8')    
    for i in range(length):
        data_[i] = image_pre(data[i],edge_len)
        label_[i] = labels[i]
    return data_,label_
    
def showHistogram(img,t1='Red',t2='Green',t3='Blue'):
    fig, axes = plt.subplots(3, 1, figsize=(5,10))
    axes[0].set_title(t1)
    axes[0].hist(img[:,:,0]);
    axes[0].set_xlim(0,255)
    axes[1].set_title(t2)
    axes[1].hist(img[:,:,1]);
    axes[1].set_xlim(0,255)
    axes[2].set_title(t3)
    axes[2].hist(img[:,:,2]);
    axes[2].set_xlim(0,255)  

def drawOneFigure(img,title=''):
    plt.figure()
    plt.axis('off')
    plt.title(title)
    plt.imshow(img)

plt.gray()
select_range = 43
edge_len = 32
channel = 1
data_shape = (channel, edge_len, edge_len)

def demonstrateLab(original_image):
    drawOneFigure(original_image,'original')
    showHistogram(original_image)
    
    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2LAB)
    drawOneFigure(image,'Lab space')
    showHistogram(image,'L','a','b')
    
    image[:,:,0] = cv2.equalizeHist(image[:,:,0])
    drawOneFigure(image,'enhanced')
    showHistogram(image,'L','a','b')
    
    convert_back = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
    drawOneFigure(convert_back,'final picture')
    showHistogram(convert_back)

def imageEnhance(img,justGetL=False):
    res = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    res[:,:,0] = cv2.equalizeHist(res[:,:,0])
    if justGetL:
        return res[:,:,0]
    else:
        return cv2.cvtColor(res, cv2.COLOR_LAB2BGR)

# data transform
if not 'train_data' in dir() or not 'train_label' in dir() or train_data == None or train_label == None:
    trainImages, trainLabels, trainClassmap = readTrafficSignsTrain('./GTSRB_train/Final_Training/Images')
    train_data,train_label = dataTransform(trainImages,trainLabels,trainClassmap)
if not 'test_data' in dir() or not 'test_label' in dir() or test_data == None or test_label == None:
    testImages, testLabels, testClassmap = readTrafficSignsTest('./GTSRB_test/Final_Test/Images')
    test_data,test_label = dataSimpleTransform(testImages,testLabels)
    valid_num = 4000
    valid_data = test_data[0:valid_num]
    valid_label = test_label[0:valid_num]
    test_data = test_data[valid_num:]
    test_label = test_label[valid_num:]


#showDataSamples(test_data,testImages)
#showDataTypes()

#demonstrateLab(testImages[11917])
#demonstrateLab(trainImages[trainClassmap[18,0]])













