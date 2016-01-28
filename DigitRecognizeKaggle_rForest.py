# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 22:22:03 2016

@author: channerduan
"""


from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt
import numpy as np

print 'load data...'
dataset = genfromtxt(open('train.csv','r'), delimiter=',', dtype='f8')[1:]
target = [x[0] for x in dataset]
train = [x[1:] for x in dataset]
test = genfromtxt(open('test.csv','r'), delimiter=',', dtype='f8')[1:]
rf = RandomForestClassifier(n_estimators=100)
print 'start train...'
rf.fit(train, target)
print 'finish train...'
res = rf.predict(test)
savetxt('submission.csv', np.c_[np.arange(1, res.size+1), res], delimiter=',', header='ImageId,Label', fmt='%d', comments='')
print 'finish result output'
