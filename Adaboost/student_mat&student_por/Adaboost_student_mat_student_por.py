#!/usr/bin/env python
# coding: utf-8

# In[191]:


import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split


# In[192]:


data1 = pd.read_csv('student_mat.csv',sep=';')
data2 = pd.read_csv('student_por.csv',sep=';')


# In[193]:


keys = ["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"]
data = pd.merge(data1,data2,on=keys)


# In[194]:


X = data.iloc[:,0:-3]
Y = data.iloc[:,-1]
X = X.values.tolist()
Y = Y.values.tolist()
#382*50


# In[195]:


for i in range(len(Y)):
    if Y[i] >= 10:
        Y[i] = 1
    else:
        Y[i] = 0


# In[196]:


#映射表 将X处理为数字
dic = {'GP':1,'MS':2,'F':1,'M':2,'U':1,'R':2,'LE3':1,'GT3':2,'T':1,'A':2,
       'at_home':1,'health':2,'services':3,'teacher':5,'other':4,'home':1,
       'reputation':2,'course':3,'mother':2,'father':3,'yes':1,'no':-1}
for i in range(len(X)):
    for j in range(len(X[i])):
        if X[i][j] in dic:
            X[i][j] = dic[X[i][j]]


# In[197]:


def buildStump(dataArr,classLabels,D):
    dataMatrix = np.mat(dataArr)
    labelMatrix = np.mat(classLabels).T
    m,n = np.shape(dataMatrix)#m = 382 n = 50
    num_steps = 10.0
    bestStump = {}
    bestClassEst = np.mat(np.zeros((m,1)))
    minError = float('inf')#据说这是无穷大
    for i in range(n):#遍历每一个属性
        rangeMin = dataMatrix[:,i].min()#找到每一个属性的最大值和最小值
        rangeMax = dataMatrix[:,i].max()
        each_step = (rangeMax-rangeMin)/num_steps
        for j in range(-1,int(num_steps)+1):
            for inequal in ['lt','gt']:
                threshVal = rangeMin+float(j)*each_step#每一步的阈值
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                error = np.mat(np.ones((m,1)))#误差矩阵
                error[predictedVals == labelMatrix] = 0#分类正确的赋值为0
                weightedError = D.T * error#
                if weightedError < minError: 									#找到误差最小的分类方式
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
                #print(minError)
    return bestStump, minError, bestClassEst


# In[198]:


def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray = np.ones((np.shape(dataMatrix)[0],1))#初始化retArray为1
    if threshIneq == 'lt':
        #print(threshVal)
        #print(dataMatrix[:,dimen])
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0	 	#如果小于阈值,则赋值为-1
        #print(retArray)
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0 		#如果大于阈值,则赋值为-1
    return retArray


# In[208]:


def adaBoostTrainDS(dataArr, classLabels, numIt = 1):#定义adaBoost
    weakClassArr = []
    m = np.shape(dataArr)[0]#行数
    D = np.mat(np.ones((m, 1)) / m)#权重矩阵
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(numIt):#相当于组合了40个弱分类器
        bestStump, error, classEst  = buildStump(dataArr,classLabels,D)
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        aggClassEst += alpha * classEst
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m,1))) 	#计算误差
        errorRate = aggErrors.sum() / m
        if errorRate == 0.0: break
    return weakClassArr, aggClassEst


# In[209]:


def adaClassify(datToClass,classifierArr):
	"""
	AdaBoost分类函数
	Parameters:
		datToClass - 待分类样例
		classifierArr - 训练好的分类器
	Returns:
		分类结果
	"""
	dataMatrix = np.mat(datToClass)
	m = np.shape(dataMatrix)[0]
	aggClassEst = np.mat(np.zeros((m,1)))
	for i in range(len(classifierArr)):										#遍历所有分类器，进行分类
		classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
		aggClassEst += classifierArr[i]['alpha'] * classEst
		#print(aggClassEst)
	return np.sign(aggClassEst)


# In[210]:


if __name__ == '__main__':
    train_X,test_X,train_Y,test_Y = train_test_split(X,Y,test_size = 0.2,random_state = 233)
    weakClassArr, aggClassEst = adaBoostTrainDS(train_X, train_Y)
    res = adaClassify(test_X, weakClassArr)
    count = 0
    for i in range(len(res)):
        if res[i]==test_Y[i]:
            count += 1
    print('正确率:%f' % (count/len(res)))


# In[ ]:





# In[ ]:
