#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 09:11:16 2018

@author: hokuanwei
"""
import numpy as np
import pandas as pd
import random


dataset = pd.read_csv('nursery.data.csv', sep=',', header=None)
print(dataset.head(5))

dataset[0] = dataset[0].replace("usual", 0)
dataset[0] = dataset[0].replace("pretentious", 1)
dataset[0] = dataset[0].replace("great_pret", 2)

dataset[1] = dataset[1].replace("proper", 0)
dataset[1] = dataset[1].replace("less_proper", 1)
dataset[1] = dataset[1].replace("improper", 2)
dataset[1] = dataset[1].replace("critical", 3)
dataset[1] = dataset[1].replace("very_crit", 4)

dataset[2] = dataset[2].replace("complete", 0)
dataset[2] = dataset[2].replace("completed", 1)
dataset[2] = dataset[2].replace("incomplete", 2)
dataset[2] = dataset[2].replace("foster", 3)

dataset[3] = dataset[3].replace("1", 0)
dataset[3] = dataset[3].replace("2", 1)
dataset[3] = dataset[3].replace("3", 2)
dataset[3] = dataset[3].replace("more", 3)

dataset[4] = dataset[4].replace("convenient", 0)
dataset[4] = dataset[4].replace("less_conv", 1)
dataset[4] = dataset[4].replace("critical", 2)

dataset[5] = dataset[5].replace("convenient", 0)
dataset[5] = dataset[5].replace("inconv", 1)

dataset[6] = dataset[6].replace("nonprob", 0)
dataset[6] = dataset[6].replace("slightly_prob", 1)
dataset[6] = dataset[6].replace("problematic", 2)

dataset[7] = dataset[7].replace("recommended", 0)
dataset[7] = dataset[7].replace("priority", 1)
dataset[7] = dataset[7].replace("not_recom", 2)

dataset[8] = dataset[8].replace("not_recom", 0)
dataset[8] = dataset[8].replace("recommend", 1)
dataset[8] = dataset[8].replace("very_recom", 2)
dataset[8] = dataset[8].replace("priority", 3)
dataset[8] = dataset[8].replace("spec_prior", 4)

x = dataset[[0, 1, 2, 3, 4, 5, 6, 7]]
y = dataset[8]

# 将数据集进行划分
'''
import random
def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]
'''

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


# 训练数据集中的样本按照类别进行划分，然后计算出每个类的统计数据
def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated


# 计算均值
import math


def mean(numbers):
    return sum(numbers) / (len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)


# 提取数据集的特征
def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries


# 按类别提取属性特征
def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries


def bayes():
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB().fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    aa = print("高斯朴素贝叶斯，样本总数： %d 預測错误样本数 : %d" % (x_test.shape[0], (y_test != y_pred).sum()))
    a = int((y_test != y_pred).sum())
    b = int(x_test.shape[0])
    bb = print('正確率', (b - a) * 100 / b)
    return aa, bb


# 计算高斯概率密度函数
def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


# 计算所属类的概率
def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.iteritems():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities


# 单一预测
def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.iteritems():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


# 多重预测
def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions


# 计算精度
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


# 主函式

def main():
    bayes()


main()
