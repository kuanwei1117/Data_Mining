贝叶斯算法分析对应数据集: 评估护士学校入学许可 nursery.data.csv

参数:
   NURSERY            Evaluation of applications for nursery schools
   . EMPLOY           Employment of parents and child's nursery
   . . parents        Parents' occupation
   . . has_nurs       Child's nursery
   . STRUCT_FINAN     Family structure and financial standings
   . . STRUCTURE      Family structure
   . . . form         Form of the family
   . . . children     Number of children
   . . housing        Housing conditions
   . . finance        Financial standing of the family
   . SOC_HEALTH       Social and health picture of the family
   . . social         Social conditions
   . . health         Health conditions

算法原理：
朴素贝叶斯方法是基于贝叶斯定理和特征条件独立假设的分类方法。对于给定的训练数据集，首先基于特征条件独立假设学习输入/输出的联合概率分布；然后基于此模型，对给定的输入x，利用贝叶斯定理求出后验概率(Maximum A Posteriori)最大的输出y。

思路分析：
给定的数据有12960组， 首先先将数据中的norminal变数转为ordinal变数，以作数据前处理及分析，接着将数据以7:3的比例随机切为训练及测试数据集，之后对训练数据集中的样本按照类别进行划分，然后计算出每个类的统计数据并计算均值，接着提取数据集的特征再按类别提取属性特征，接着进入贝叶斯算法分析的部分，先计算高斯概率密度函数并计算所属类的概率，接着对变数进行单一预测与多重预测，最后计算精度，并打印出预测结果与准确率。

测试结果：
从输出结果的正确率来看，此时训练数据9072，测试数据3888，预测错误样本数为951，正确率为75.54%
