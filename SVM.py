#import libraries
from numpy import genfromtxt, savetxt
import numpy as np
from sklearn import svm
from sklearn import preprocessing

#accuracy function
def accuracy(target, result):
    count=0;
    for i in range(len(target)):
        if(result[i]==target[i]):
            count= count+1
    return(count*100.0/len(target))

#combining classifiers with weightage
def combine(res1, res2, rat1, rat2):
    for i in range(len(res1)):
        rnd=random.random()
        res=[]
        if(res1[i]!=res2[i]):
            if(rnd>(rat1/(rat1+rat2))):
                res.append(res2[i])
                print str(res1[i])+" "+str(res2[i])+" "+str(res2[i])
            else:
                res.append(res1[i])
                print str(res1[i])+" "+str(res2[i])+" "+str(res1[i])
        else:
            res.append(res1[i])
    return res

#load dataset
dataset = genfromtxt(open('train.csv','r'), delimiter=',', dtype='f8')[1:]
target = [x[0] for x in dataset]
train = [x[1:] for x in dataset]
test = genfromtxt(open('test.csv','r'), delimiter=',', dtype='f8')[1:]

noPoints=2000 #set to any number
train2=train[0:noPoints]
target2=target[0:noPoints]
scaler = preprocessing.StandardScaler().fit(train2)
train2=scaler.transform(train2)
test2=scaler.transform(test)

#polynomial kernel, degree=3
clf3=svm.SVC(C=100.0,kernel='poly')
clf3.fit(scaler.transform(train2),target)
res=clf3.predict(scaler.transform(test))

print res

savetxt('submissionPoly.csv', np.c_[np.arange(1, res.size+1), res], delimiter=',', header='ImageId,Label', fmt='%d', comments='')

#for tuning RBF kernel SVM
#C=2444.4444444444443
#gamma=5.9948425031894092e-07
results=[]
C_range = np.logspace(-2, 10, 10) #keep changing
gamma_range = np.logspace(-9, 3, 10) #keep changing
for C in C_range:
    for gamma in gamma_range:
        clf = svm.SVC(C=C, gamma=gamma)
        clf.fit(train2,target2)
        res=clf.predict(test2)
        results.append(res)

accs=[]
for i in range(0,len(results)):
    accs.append(accuracy(results[i],rfres))










