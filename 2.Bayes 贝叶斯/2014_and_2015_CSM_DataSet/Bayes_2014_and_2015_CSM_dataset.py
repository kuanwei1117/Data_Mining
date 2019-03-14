#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 19:43:57 2018

@author: hokuanwei
"""

import numpy as np
import pandas as pd
from math import sqrt
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_excel('2014_and_2015_CSM_dataset.xlsx',sep=',')
print(dataset.head(5))

#sns.pairplot(dataset, x_vars=['Genre','Budget','Screens','Sequel','Sentiment','Views','Likes','Dislikes','Comments','Aggregate Followers'], y_vars='Ratings', size=7, aspect=0.8,kind='reg')
#plt.show()

#create a python list of feature names
feature_cols = ['Genre','Budget','Screens','Sequel','Sentiment','Views','Likes','Dislikes','Comments','Aggregate Followers']
# use the list to select a subset of the original DataFrame
X = dataset[feature_cols]
# equivalent command to do this in one line
X = dataset[['Genre','Budget','Screens','Sequel','Sentiment','Views','Likes','Dislikes','Comments','Aggregate Followers']]
# print the first 5 rows
print (X.head())
# check the type and shape of X
print (type(X))
print (X.shape)


# select a Series from the DataFrame
y = dataset['Ratings']
# equivalent command that works if there are no spaces in the column name
y = dataset.Ratings
# print the first 5 values
print (y.head())

from sklearn.cross_validation import train_test_split  #这里是引用了交叉验证
X_train,X_test, y_train, y_test = train_test_split(X, y, random_state=1)

print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
model=linreg.fit(X_train, y_train)
print (model)
print (linreg.intercept_)

for i in range(10):
  print(feature_cols[i],linreg.coef_[i])

#預測
y_pred = linreg.predict(X_test)
print (y_pred)

#評價測度
print (type(y_pred),type(y_test))
print (len(y_pred),len(y_test))
print (y_pred.shape,y_test.shape)
from sklearn import metrics
import numpy as np
sum_mean=0
for i in range(len(y_pred)):
    sum_mean+=(y_pred[i]-y_test.values[i])**2
sum_erro=np.sqrt(sum_mean/50)
# calculate RMSE by hand
print ("RMSE by hand:",sum_erro)

#做ROC曲线
import matplotlib.pyplot as plt
plt.figure()
plt.plot(range(len(y_pred)),y_pred,'b',label="predict")
plt.plot(range(len(y_pred)),y_test,'r',label="test")
plt.legend(loc="upper right") #显示图中的标签
plt.xlabel("the number of sales")
plt.ylabel('value of sales')
plt.show()
