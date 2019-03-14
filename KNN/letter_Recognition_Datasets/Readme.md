采用Knn算法分析letter Recognition Datasets数据集

算法原理：
给定一个训练数据集，对新的输入实例，在训练数据集中找到与该实例最邻近的 k 个实例，这 k 个实例的多数属于某个类，就把该输入实例分为这个类。

思路分析：
给定的数据有20000组，将训练数据的数量和k的取值作为两个变量，调节这两个变量，尽量获得高的预测正确率。首先读挡做资料前处理，得出变量特征矩阵以及分类标签向量。之后送入分类器计算训练数据的KNN算法以各笔得出之分类类别与空间中其他笔数据之欧基里得距离后,比较空间中其他笔数据的类别, 预测该笔数据之标签类别,再将测试数据放入训练好的模型，print出分类结果与正确率。

测试结果：
从输出结果的正确率来看，k=5为k的最佳取值，此时训练数据16000，测试数据2000，正确率为95.75%