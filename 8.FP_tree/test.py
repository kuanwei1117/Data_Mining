#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 11:32:41 2018

@author: hokuanwei
"""
import fpgrowth
import pandas as pd
import numpy as np

"""導入數據"""
df_temp = pd.read_csv("retail.csv", sep=',', error_bad_lines=False)

df_temp = df_temp.values

"""處理數據"""
df = []
for i in range(len(df_temp)):
    temp = df_temp[i]
    r = temp[np.logical_not(np.isnan(temp))]
    df.append(r.tolist())

label = []
items = []
for i in range(len(df)):
    for j in range(len(df[i])):
        if df[i][j] not in label:
            label.append(df[i][j])
            items.append((int(df[i][j])))

initSet = fpgrowth.createInitSet(df)
# 用数据集构造FP树，最小支持度10w
myFPtree, myHeaderTab = fpgrowth.createFPtree(initSet, 2000)

# 挖掘FP树
freqItems = []
fpgrowth.mineFPtree(myFPtree, myHeaderTab, 2000, set([]), freqItems)
for x in freqItems:
    print (x)
