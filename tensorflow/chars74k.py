#!/usr/bin/env python
# encoding: utf-8
"""
chars74k.py

Created by Darcy on 10/04/2016.
Copyright (c) 2016 Darcy. All rights reserved.
"""

import sys
import os
import time
import urllib
import tarfile
import numpy as np
import pandas as pd
from scipy import ndimage, misc
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

SOURCE_URL = 'http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/'
WORK_DIRECTORY = 'data'


def raadimg(path, character, num):
    folder = 'Sample%03d' % (character, )
    sample = os.path.join(path, folder)
    filename = os.path.join(sample, 'img%03d-%03d.png' % (character, num))
    if not os.path.exists(filename):
        print filename
    img = ndimage.imread(filename, flatten=True)
    img = misc.imresize(img, (30, 40))
    w, h = img.shape
    # print w, h
    # print img.reshape((w*h))
    return img.reshape((w*h))


def extract_data():
    if os.path.exists('training_data_mixed.csv'):
        return

    filename = 'EnglishHnd.tgz'
    if not os.path.exists(WORK_DIRECTORY):
        os.mkdir(WORK_DIRECTORY)

    path = os.path.join(WORK_DIRECTORY, filename)
    if not os.path.exists(path):
        def fn(count, block_size, total_size):
            progress = 100.0
            if total_size < block_size * count:
                pass
            else:
                progress = float(count * block_size) / float(total_size) * 100.0
            sys.stdout.write('>> Downloading %s %.1f%%\n' % (filename, progress))
            sys.stdout.flush()
        file_path, _ = urllib.urlretrieve(SOURCE_URL + filename, path, reporthook=fn)
        stat_info = os.stat(file_path)
        print 'Successfully downloaded', filename, stat_info.st_size, 'bytes.'
        tar = tarfile.open(path, 'r')
        for item in tar:
            tar.extract(item, WORK_DIRECTORY)

    path = os.path.join(WORK_DIRECTORY, 'English/Hnd/Img/')

    training_data = []
    training_label = []
    for i in range(0, 10):
        print i+1, i
        for j in range(0, 55):
            img = raadimg(path, i+1, j+1)
            training_data.append(img)
            training_label.append(i+1)

    for i in range(0, 26):
        print 11 + i, chr(ord('A') + i)

        for j in range(0, 55):
            img = raadimg(path, i+11, j+1)
            training_data.append(img)
            training_label.append(i+1)

    for i in range(0, 26):
        print 37 + i, chr(ord('a') + i)
        for j in range(0, 55):
            img = raadimg(path, i+37, j+1)
            training_data.append(img)
            training_label.append(i+1)

    training_data = np.array(training_data).astype(np.uint8)
    np.savetxt('training_data.csv', np.c_[training_label, training_data], delimiter=',', fmt ='%d')

    training_data = []
    training_label = []
    for j in range(0, 55):
        for i in range(0, 10):
            img = raadimg(path, i+1, j+1)
            training_data.append(img)
            training_label.append(i+1)
        for i in range(0, 26):
            img = raadimg(path, i+11, j+1)
            training_data.append(img)
            training_label.append(i+1)
        for i in range(0, 26):
            img = raadimg(path, i+37, j+1)
            training_data.append(img)
            training_label.append(i+1)

    training_data = np.array(training_data).astype(np.uint8)
    np.savetxt('training_data_mixed.csv', np.c_[training_label, training_data], delimiter=',', fmt ='%d')


def training(t):
    dataset = pd.read_csv('training_data_mixed.csv', header=None).values
    # print 'dataset: ', dataset.shape

    labels = dataset[:, 0]
    images = dataset[:, 1:].astype(np.float32)
    images = np.multiply(images, 1.0 / 255.0)
    # print labels.shape
    # print images.shape

    train_labels = labels[:62*50]
    train_images = images[:62*50]
    # print train_labels.shape
    # print train_images.shape

    test_labels = labels[62*50:62*55]
    test_images = images[62*50:62*55]

    train_labels = input_data.dense_to_one_hot(train_labels, 62)
    test_labels = input_data.dense_to_one_hot(test_labels, 62)

    # print train_labels.shape
    # print test_labels.shape
    x = tf.placeholder(tf.float32, [None, 30*40])

    W = tf.Variable(tf.zeros([30*40, 62]))
    b = tf.Variable(tf.zeros([62]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    y_ = tf.placeholder(tf.float32, [None, 62])
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))

    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    start = 0
    for i in range(t):
        end = start + 62
        batch_xs = train_images[start:end, ]
        batch_ys = train_labels[start:end, ]

        # print batch_xs.shape
        # print batch_ys.shape
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        start = end
        if (start+62) > len(train_images):
            start = 0

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    print 'accuracy: ', sess.run(accuracy, feed_dict={x: test_images, y_: test_labels})


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def cnn():
    dataset = pd.read_csv('training_data_mixed.csv', header=None).values
    # print 'dataset: ', dataset.shape

    labels = dataset[:, 0]
    images = dataset[:, 1:].astype(np.float32)
    images = np.multiply(images, 1.0 / 255.0)
    # print labels.shape
    # print images.shape

    train_labels = labels[:62*50]
    train_images = images[:62*50]
    # print train_labels.shape
    # print train_images.shape

    test_labels = labels[62*50:62*55]
    test_images = images[62*50:62*55]

    train_labels = input_data.dense_to_one_hot(train_labels, 62)
    test_labels = input_data.dense_to_one_hot(test_labels, 62)

    x = tf.placeholder("float", shape=[None, 30*40])
    y_ = tf.placeholder("float", shape=[None, 62])
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1,30,40,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([8 * 10 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 8*10*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 62])
    b_fc2 = bias_variable([62])

    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())

    start = 0
    for i in range(2000):
        end = start + 62
        batch_xs = train_images[start:end, ]
        batch_ys = train_labels[start:end, ]
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:test_images, y_: test_labels, keep_prob: 1.0})
            print "step %d, training accuracy %g"%(i, train_accuracy)
        train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

        start = end
        if (start+62) > len(train_images):
            start = 0

    print "test accuracy %g" % accuracy.eval(feed_dict={x: test_images, y_: test_labels, keep_prob: 1.0})


def main():
    start = time.time()
    extract_data()

    # for i in range(500, 10000, 500):
    #     print i
    #     training(i)
    cnn()
    end = time.time()
    print 'CPU time: %f seconds.' % (end - start,)


if '__main__' == __name__:
    main()
