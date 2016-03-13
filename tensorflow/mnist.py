#!/usr/bin/env python
# encoding: utf-8
"""
mnist.py

Created by Darcy on 06/02/2016.
Copyright (c) 2016 Darcy. All rights reserved.
"""

import sys
import os
import time
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def simple():
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})


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


def mnist_pros():
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

    x = tf.placeholder("float", shape=[None, 784])
    y_ = tf.placeholder("float", shape=[None, 10])

    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    y = tf.nn.softmax(tf.matmul(x,W) + b)
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    for i in range(1000):
        batch = mnist.train.next_batch(50)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})


def conv():
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

    x = tf.placeholder("float", shape=[None, 784])
    y_ = tf.placeholder("float", shape=[None, 10])
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1,28,28,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
            print "step %d, training accuracy %g"%(i, train_accuracy)
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print "test accuracy %g" % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})


def cnn():
    NUM_ITERAtION = 25000
    VALIDATION_SIZE = 2000
    IMAGE_TO_DISPLAY = 10

    # read test data from CSV file
    test_images = pd.read_csv('../mnist_raw/test.csv').values
    test_images = test_images.astype(np.float)
    test_images = (test_images - (255.0 / 2.0)) / 255.0

    print('test_images({0[0]},{0[1]})'.format(test_images.shape))

    test_labels = np.loadtxt('submission_test.csv', np.int32,  delimiter=',', skiprows=1)
    test_labels = test_labels[:, 1]
    print(test_labels.shape)

    data = pd.read_csv('../mnist_raw/train.csv')

    images = data.iloc[:, 1:].values
    images = images.astype(np.float)
    images = (images - (255.0 / 2.0)) / 255.0

    print('images({0[0]},{0[1]})'.format(images.shape))

    image_size = images.shape[1]
    print ('image_size => {0}'.format(image_size))

    # in this case all images are square
    image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)

    print ('image_width => {0}\nimage_height => {1}'.format(image_width,image_height))

    labels_flat = data[[0]].values.ravel()

    print('labels_flat({0})'.format(len(labels_flat)))
    print ('labels_flat[{0}] => {1}'.format(IMAGE_TO_DISPLAY,labels_flat[IMAGE_TO_DISPLAY]))

    labels_count = np.unique(labels_flat).shape[0]

    print('labels_count => {0}'.format(labels_count))

    labels = input_data.dense_to_one_hot(labels_flat, labels_count)
    labels = labels.astype(np.uint8)
    print('labels({0[0]},{0[1]})'.format(labels.shape))
    print ('labels[{0}] => {1}'.format(IMAGE_TO_DISPLAY,labels[IMAGE_TO_DISPLAY]))

    # split data into training & validation
    validation_images = images[:VALIDATION_SIZE]
    validation_labels = labels[:VALIDATION_SIZE]

    train_images = images[VALIDATION_SIZE:]
    train_labels = labels[VALIDATION_SIZE:]

    print('train_images({0[0]},{0[1]})'.format(train_images.shape))
    print('validation_images({0[0]},{0[1]})'.format(validation_images.shape))

    x = tf.placeholder("float", shape=[None, 784])
    y_ = tf.placeholder("float", shape=[None, 10])

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1,28,28,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())

    start = 0
    for i in range(NUM_ITERAtION):
        end = start+50
        batch_xs = train_images[start:end]
        batch_ys = train_labels[start:end]
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:validation_images, y_: validation_labels, keep_prob: 1.0})
            print "step %d, training accuracy %g"%(i, train_accuracy)

        if i%1000 == 0:
            predicted_lables = accuracy.eval(feed_dict={x: test_images, keep_prob: 1.0})
            error_rate = 100.0 - (100.0 * np.sum(predicted_lables == test_labels) / predicted_lables.shape[0])
            
            print 'error_rate', error_rate

            np.savetxt('data/submission_cnn_%d.csv' % i,
                np.c_[range(1,len(predicted_lables)+1), predicted_lables],
                delimiter=',',
                header = 'ImageId,Label',
                comments = '',
                fmt ='%d')    
        train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

        start = end
        if (start + 50) > len(train_images):
            start = 0

    predict = tf.argmax(y_conv, 1)
    predicted_lables = predict.eval(feed_dict={x: test_images, keep_prob: 1.0})

    print(predicted_lables.shape)
    error_rate = 100.0 - (100.0 * np.sum(predicted_lables == test_labels) / predicted_lables.shape[0])
    print('error_rate', error_rate)

    np.savetxt('submission_cnn.csv',
               np.c_[range(1,len(predicted_lables)+1), predicted_lables],
               delimiter=',',
               header = 'ImageId,Label',
               comments = '',
               fmt ='%d')


def summary():
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    W = tf.Variable(tf.zeros([784, 10]), name='weights')
    b = tf.Variable(tf.zeros([10]), name='bias')
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    _ = tf.scalar_summary("cross entropy", cross_entropy)

    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    _ = tf.scalar_summary('accuracy', accuracy)

    # Add summary ops to collect data
    _ = tf.histogram_summary('weights', W)
    _ = tf.histogram_summary('biases', b)
    _ = tf.histogram_summary('y', y)

    _ = tf.image_summary('images', tf.reshape(mnist.test.images, [10000,28,28,1]))

    merged = tf.merge_all_summaries()
    saver = tf.train.Saver()

    init = tf.initialize_all_variables()
    sess = tf.Session()

    writer = tf.train.SummaryWriter('logs', sess.graph_def)

    sess.run(init)

    for step in range(2000):
        if step % 10 == 0:
            saver.save(sess, 'logs', global_step=step)
            feed = {x: mnist.test.images, y_: mnist.test.labels}
            summary_str, acc = sess.run([merged, accuracy], feed_dict=feed)
            writer.add_summary(summary_str, step)
            print('Accuracy at step %s: %s' % (step, acc))
        else:
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})


def checkpoint(train=True):
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    W = tf.Variable(tf.zeros([784, 10]), name='weights')
    b = tf.Variable(tf.zeros([10]), name='bias')
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    _ = tf.scalar_summary("cross entropy", cross_entropy)

    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    _ = tf.scalar_summary('accuracy', accuracy)

    # Add summary ops to collect data
    _ = tf.histogram_summary('weights', W)
    _ = tf.histogram_summary('biases', b)
    _ = tf.histogram_summary('y', y)

    _ = tf.image_summary('images', tf.reshape(mnist.test.images, [10000, 28, 28, 1]))

    merged = tf.merge_all_summaries()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)

        if train:
            writer = tf.train.SummaryWriter('logs', sess.graph_def)
            for step in range(1000):
                if step % 10 == 0:
                    saver.save(sess, 'logs/checkpoint', global_step=step)
                    feed = {x: mnist.test.images, y_: mnist.test.labels}
                    summary_str, acc = sess.run([merged, accuracy], feed_dict=feed)
                    writer.add_summary(summary_str, step)
                    print('Accuracy at step %s: %s' % (step, acc))
                else:
                    batch_xs, batch_ys = mnist.train.next_batch(100)
                    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

            print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        else:
            check_point = tf.train.get_checkpoint_state('logs')
            if checkpoint and check_point.model_checkpoint_path:
                saver.restore(sess, check_point.model_checkpoint_path)
                print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
            else:
                print 'error'


def maxnet(train=True, logdir='logs', iterations=100):
    # mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

    VALIDATION_SIZE = 2000
    IMAGE_TO_DISPLAY = 10

    # read test data from CSV file
    test_images = pd.read_csv('../mnist_raw/test.csv').values
    test_images = test_images.astype(np.float32)
    test_images = (test_images - (255.0 / 2.0)) / 255.0

    print('test_images({0[0]},{0[1]})'.format(test_images.shape))

    test_labels = np.loadtxt('submission_test.csv', np.int32,  delimiter=',', skiprows=1)
    test_labels = test_labels[:, 1]

    test_labels = input_data.dense_to_one_hot(test_labels, 10)
    test_labels = test_labels.astype(np.uint8)

    print(test_labels.shape)

    data = pd.read_csv('../mnist_raw/train.csv')

    images = data.iloc[:, 1:].values
    images = images.astype(np.float32)
    images = (images - (255.0 / 2.0)) / 255.0

    print('images({0[0]},{0[1]})'.format(images.shape))

    image_size = images.shape[1]
    print ('image_size => {0}'.format(image_size))

    # in this case all images are square
    image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)

    print ('image_width => {0}\nimage_height => {1}'.format(image_width,image_height))

    labels_flat = data[[0]].values.ravel()

    print('labels_flat({0})'.format(len(labels_flat)))
    print ('labels_flat[{0}] => {1}'.format(IMAGE_TO_DISPLAY, labels_flat[IMAGE_TO_DISPLAY]))

    labels_count = np.unique(labels_flat).shape[0]

    print('labels_count => {0}'.format(labels_count))

    labels = input_data.dense_to_one_hot(labels_flat, labels_count)
    labels = labels.astype(np.uint8)
    print('labels({0[0]},{0[1]})'.format(labels.shape))
    print ('labels[{0}] => {1}'.format(IMAGE_TO_DISPLAY,labels[IMAGE_TO_DISPLAY]))

    # split data into training & validation
    validation_images = images[:VALIDATION_SIZE]
    validation_labels = labels[:VALIDATION_SIZE]

    train_images = images[VALIDATION_SIZE:]
    train_labels = labels[VALIDATION_SIZE:]

    print('train_images({0[0]},{0[1]})'.format(train_images.shape))
    print('validation_images({0[0]},{0[1]})'.format(validation_images.shape))

    x = tf.placeholder("float", shape=[None, 784])
    y_ = tf.placeholder("float", shape=[None, 10])
    W_conv1 = weight_variable([5, 5, 1, 20])
    b_conv1 = bias_variable([20])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.tanh(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    W_conv2 = weight_variable([5, 5, 20, 40])
    b_conv2 = bias_variable([40])

    h_conv2 = tf.tanh(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 40, 150])
    b_fc1 = bias_variable([150])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 40])
    h_fc1 = tf.tanh(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([150, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    _ = tf.scalar_summary('accuracy', accuracy)

    # Add summary ops to collect data
    _ = tf.histogram_summary('W_conv1', W_conv1)
    _ = tf.histogram_summary('b_conv1', b_conv1)

    _ = tf.histogram_summary('W_conv2', W_conv2)
    _ = tf.histogram_summary('b_conv2', b_conv2)

    _ = tf.histogram_summary('W_fc1', W_fc1)
    _ = tf.histogram_summary('b_fc1', b_fc1)

    _ = tf.histogram_summary('W_fc2', W_fc2)
    _ = tf.histogram_summary('b_fc2', b_fc2)

    _ = tf.histogram_summary('y_', y_)

    _ = tf.image_summary('images', tf.reshape(test_images, [test_images.shape[0], 28, 28, 1]))

    merged = tf.merge_all_summaries()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        if train:
            writer = tf.train.SummaryWriter(logdir, sess.graph_def)

            start = 0
            for step in range(iterations):
                # batch = mnist.train.next_batch(50)
                end = start+50
                batch_xs = train_images[start:end]
                batch_ys = train_labels[start:end]
                if step % 100 == 0:
                    saver.save(sess, os.path.join(logdir, 'checkpoint'), global_step=step)
                    feed = {x: validation_images, y_: validation_labels, keep_prob: 1.0}
                    summary_str, acc = sess.run([merged, accuracy], feed_dict=feed)
                    writer.add_summary(summary_str, step)
                    print('Accuracy at step %s: %s' % (step, acc))

                if (iterations - 1) == step:
                    saver.save(sess, os.path.join(logdir, 'checkpoint'), global_step=step)

                train_step.run(feed_dict={x: batch_xs, y_:batch_ys, keep_prob: 0.5})

            print "test accuracy %g" % accuracy.eval(feed_dict={
                x: test_images, y_: test_labels, keep_prob: 1.0})
        else:
            check_point = tf.train.get_checkpoint_state(logdir)
            print check_point
            if checkpoint and check_point.model_checkpoint_path:
                saver.restore(sess, check_point.model_checkpoint_path)
                print sess.run(accuracy, feed_dict={x: test_images, y_: test_labels, keep_prob: 1.0})
            else:
                print 'error'


def main():
    parser = argparse.ArgumentParser(description='TensorFlow demo')
    parser.add_argument('-train', action='store_true',  default=False, help='train flag')
    parser.add_argument('-logdir', default='logs', help='log dir')
    parser.add_argument('-i', dest='iterations', type=int, default=100, help='iterations')
    args = parser.parse_args()

    start = time.time()
    # checkpoint(args.train)
    maxnet(train=args.train, logdir=args.logdir, iterations=args.iterations)
    end = time.time()
    print 'CPU time is %f seconds.' % (end - start)


if '__main__' == __name__:
    main()
