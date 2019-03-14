
# coding: utf-8

from math import log
import operator
import treePlotter
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


if __name__ == '__main__':
    with open('page-blocks＿corr.txt','r') as fr:
        dataSet = [inst.strip().split() for inst in fr.readlines()]
    target = []
    for each in dataSet:
        target.append(each[-1])

    labels = ['height', 'lenght', 'area', 'eccen','p_black','p_and','mean_tr','blackpix','blackand','wb_trans']


def calcShannonEnt(dataSet):
    """
    输入：数据集
    输出：数据集的香农熵
    描述：计算给定数据集的香农熵；熵越大，数据集的混乱程度越大
    """
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt



def majorityCnt(classList):
    """
    输入：分类类别列表
    输出：子节点的分类
    描述：数据集已经处理了所有属性，但是类标签依然不是唯一的，
          采用多数判决的方法决定该子节点的分类
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reversed=True)
    return sortedClassCount[0][0]



def chooseBestFeatureToSplit(dataSet):
    """
    输入：数据集
    输出：最好的划分维度
    描述：选择最好的数据集划分维度
    """
    numFeatures = len(dataSet[0]) - 1                 #feature个数
    baseEntropy = calcShannonEnt(dataSet)             #整个dataset的熵
    bestInfoGainRatio = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]  #每个feature的list
        uniqueVals = set(featList)                      #每个list的唯一值集合
        newEntropy = 0.0
        splitInfo = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)  #每个唯一值对应的剩余feature的组成子集
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
            splitInfo += -prob * log(prob, 2)
        infoGain = baseEntropy - newEntropy              #这个feature的infoGain
        if (splitInfo == 0): # fix the overflow bug
            continue
        infoGainRatio = infoGain / splitInfo             #这个feature的infoGainRatio
        if (infoGainRatio > bestInfoGainRatio):          #选择最大的gain ratio
            bestInfoGainRatio = infoGainRatio
            bestFeature = i                              #选择最大的gain ratio对应的feature
    return bestFeature



def splitDataSet(dataSet, axis, value):
    """
    输入：数据集，选择维度，选择值
    输出：划分数据集
    描述：按照给定特征划分数据集；去除选择维度中等于选择值的项
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:                      #只看当第i列的值＝value时的item
            reduceFeatVec = featVec[:axis]              #featVec的第i列给除去
            reduceFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet



def createTree(dataSet, labels):
    """
    输入：数据集，特征标签
    输出：决策树
    描述：递归构建决策树，利用上述的函数
    """
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        # classList所有元素都相等，即类别完全相同，停止划分
        return classList[0]                                  #splitDataSet(dataSet, 0, 0)此时全是N，返回N
    if len(dataSet[0]) == 1:
        # 遍历完所有特征时返回出现次数最多的
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)             #0－> 2
        # 选择最大的gain ratio对应的feature
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
            # 划分数据，为下一层计算准备
    return myTree



labels_tmp = labels[:]
desicionTree = createTree(dataSet, labels_tmp)
treePlotter.createPlot(desicionTree)




def classify(inputTree, featLabels, testVec):
    """
    输入：决策树，分类标签，测试数据
    输出：决策结果
    描述：跑决策树
    """

    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':       # type(secondDict[key]).__name__＝str
                # 如果secondDict[key]仍然是字典，则继续向下层走
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                # 如果secondDict[key]已经只是分类标签了，则返回这个类别标签
                classLabel = secondDict[key]
    return classLabel



def createTestSet():

    testSet=[]
    i=20
    for i in range(120):
        testSet.append(dataSet[i])
    return testSet

inputTree = desicionTree
featLabels = ['height', 'lenght', 'area', 'eccen','p_black','p_and','mean_tr','blackpix','blackand','wb_trans']
testVec = dataSet[0]
classify(inputTree, featLabels, testVec)




def classifyAll(inputTree, featLabels, testDataSet):
    """
    输入：决策树，分类标签，测试数据集
    输出：决策结果
    描述：跑决策树
    """
    classLabelAll = []
    for testVec in testDataSet:               #逐个item进行分类判断
        classLabelAll.append(classify(inputTree, featLabels, testVec))
    return classLabelAll




testSet = createTestSet()
print('classifyResult:\n', classifyAll(desicionTree, labels, testSet))




accuracy_arr = []
predicte_arr = []
for i in range(1000):
    predicte_arr.append(classify(inputTree, featLabels, dataSet[i]))
    accuracy_arr.append(dataSet[i][10])

print(accuracy_metric(accuracy_arr, predicte_arr))




def main():
    labels_tmp = labels[:]
    desicionTree = createTree(dataSet, labels_tmp)
    treePlotter.createPlot(desicionTree)

if __name__ == '_main_':
    main()




def accuracy_metric(actual, predicted):
 correct = 0
 for i in range(len(actual)):
  if actual[i] == predicted[i]:
   correct += 1
 return correct / float(len(actual)) * 100.0
