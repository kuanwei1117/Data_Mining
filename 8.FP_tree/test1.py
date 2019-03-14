#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 12:18:45 2018

@author: hokuanwei
"""
import fpgrowth
# 数据集
def loadSimpDat():
    simDat = [['apple','beer','rice','chicken'],
              ['apple','beer','rice'],
              ['apple','beer'],
              ['apple','mango'],
              ['milk','beer','rice','chicken'],
              ['milk','beer','rice'],
              ['milk','beer'],
              ['milk','mango']]
    
    return simDat
# 构造成 element : count 的形式
def createInitSet(dataSet):
    retDict={}
    for trans in dataSet:
        key = frozenset(trans)
        if key in retDict:
            retDict[frozenset(trans)] += 1
        else:
            retDict[frozenset(trans)] = 1
    return retDict

simDat = loadSimpDat()
initSet = createInitSet(simDat)
myFPtree, myHeaderTab = fpgrowth.createFPtree(initSet, 3)

freqItems = []
fpgrowth.mineFPtree(myFPtree, myHeaderTab, 3, set([]), freqItems)
for x in freqItems:
    print (x)
