# -*- coding: utf-8 -*-
from __future__ import division
import operator
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



#KNN分析
def classify0(inX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]
	diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances**0.5
	sortedDistIndices = distances.argsort()
	classCount = {}
	for i in range(k):
		voteIlabel = labels[sortedDistIndices[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
	sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
	return sortedClassCount[0][0]

data = np.loadtxt("letter-recognition_data.txt",dtype=str,delimiter=",")
yy, x = np.split(data, (1,), axis=1)
#print yy.shape, x.shape
#print x
#print yy

#將str轉成int

data = x.astype(int)
label = np.ravel(yy)




def autoNorm(dataSet):
	minVals = dataSet.min(axis=0)
	maxVals = dataSet.max(axis=0)
	ranges = maxVals - minVals
	normDataSet = (dataSet-minVals)/ranges
	return normDataSet, ranges, minVals

b,c,d = autoNorm(data)
print(b,c,d)

def testclass():
    hoRatio=0.2
    normMat,ranges,minVals=autoNorm(data)
    m=normMat.shape[0]
    numTestVecs=int(m*hoRatio)

    errorCount=0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:],label[numTestVecs:m], 5)
        print('預測結果：{},實際結果：{}'.format(classifierResult, label[i]))
        if classifierResult != label[i]:
            errorCount += 1.0
    a = 1.0-errorCount/float(numTestVecs)
    print('k=',5,'時\t',"正確率",a)


testclass()
