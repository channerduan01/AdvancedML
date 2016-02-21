# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 22:22:03 2016

@author: channerduan
"""
import sys
sys.path.append("../common")
from timer import Timer
from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt
import numpy as np


with Timer() as t:
    print 'load data...'
    dataset = genfromtxt(open('../mnist_raw/train.csv','r'), delimiter=',', dtype='f8')[1:]
    target = [x[0] for x in dataset]
    train = [x[1:] for x in dataset]
    test = genfromtxt(open('../mnist_raw/test.csv','r'), delimiter=',', dtype='f8')[1:]
    rf = RandomForestClassifier(n_estimators=100)
print "=> load data: %s s" % t.secs

with Timer() as t:
    print 'start train...'
    rf.fit(train, target)
    print 'finish train...'
print "=> train: %s s" % t.secs

with Timer() as t:
    res = rf.predict(test)
print "=> prediction: %s s" % t.secs

savetxt('submission.csv', np.c_[np.arange(1, res.size+1), res], delimiter=',', header='ImageId,Label', fmt='%d', comments='')
print 'finish result output'
