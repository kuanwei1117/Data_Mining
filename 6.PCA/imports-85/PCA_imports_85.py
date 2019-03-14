
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('/Users/hokuanwei/Desktop/第6次作业/PCA数据集/imports-85.data.csv', sep=',', header=None)
# print(dataset.head(5))
dataset[0] = dataset[0].replace("?", 0)

dataset[1] = dataset[1].replace("?", 0)

dataset[2] = dataset[2].replace("alfa-romero", 1)
dataset[2] = dataset[2].replace("audi", 2)
dataset[2] = dataset[2].replace("bmw", 3)
dataset[2] = dataset[2].replace("chevrolet", 4)
dataset[2] = dataset[2].replace("dodge", 5)
dataset[2] = dataset[2].replace("honda", 6)
dataset[2] = dataset[2].replace("isuzu", 7)
dataset[2] = dataset[2].replace("jaguar", 8)
dataset[2] = dataset[2].replace("mazda", 9)
dataset[2] = dataset[2].replace("mercedes-benz", 10)
dataset[2] = dataset[2].replace("mercury", 11)
dataset[2] = dataset[2].replace("mitsubishi", 12)
dataset[2] = dataset[2].replace("nissan", 13)
dataset[2] = dataset[2].replace("peugot", 14)
dataset[2] = dataset[2].replace("plymouth", 15)
dataset[2] = dataset[2].replace("porsche", 16)
dataset[2] = dataset[2].replace("mercedes-benz", 10)
dataset[2] = dataset[2].replace("renault", 17)
dataset[2] = dataset[2].replace("saab", 18)
dataset[2] = dataset[2].replace("subaru", 19)
dataset[2] = dataset[2].replace("toyota", 20)
dataset[2] = dataset[2].replace("volkswagen", 21)
dataset[2] = dataset[2].replace("volvo", 22)

dataset[3] = dataset[3].replace("?", 0)
dataset[3] = dataset[3].replace("diesel", 1)
dataset[3] = dataset[3].replace("gas", 2)

dataset[4] = dataset[4].replace("?", 0)
dataset[4] = dataset[4].replace("std", 1)
dataset[4] = dataset[4].replace("turbo", 2)

dataset[5] = dataset[5].replace("?", 0)
dataset[5] = dataset[5].replace("four", 1)
dataset[5] = dataset[5].replace("two", 2)

dataset[6] = dataset[6].replace("?", 0)
dataset[6] = dataset[6].replace("hardtop", 1)
dataset[6] = dataset[6].replace("wagon", 2)
dataset[6] = dataset[6].replace("sedan", 3)
dataset[6] = dataset[6].replace("hatchback", 4)
dataset[6] = dataset[6].replace("convertible", 5)

dataset[7] = dataset[7].replace("?", 0)
dataset[7] = dataset[7].replace("4wd", 1)
dataset[7] = dataset[7].replace("fwd", 2)
dataset[7] = dataset[7].replace("rwd", 3)

dataset[8] = dataset[8].replace("?", 0)
dataset[8] = dataset[8].replace("front", 1)
dataset[8] = dataset[8].replace("rear", 2)

dataset[9] = dataset[9].replace("?", 0)

dataset[10] = dataset[10].replace("?", 0)

dataset[11] = dataset[11].replace("?", 0)

dataset[12] = dataset[12].replace("?", 0)

dataset[13] = dataset[13].replace("?", 0)

dataset[14] = dataset[14].replace("?", 0)
dataset[14] = dataset[14].replace("dohc", 1)
dataset[14] = dataset[14].replace("dohcv", 2)
dataset[14] = dataset[14].replace("l", 3)
dataset[14] = dataset[14].replace("ohc", 4)
dataset[14] = dataset[14].replace("ohcf", 5)
dataset[14] = dataset[14].replace("ohcv", 6)
dataset[14] = dataset[14].replace("rotor", 7)

dataset[15] = dataset[15].replace("?", 0)
dataset[15] = dataset[15].replace("eight", 1)
dataset[15] = dataset[15].replace("five", 2)
dataset[15] = dataset[15].replace("four", 3)
dataset[15] = dataset[15].replace("six", 4)
dataset[15] = dataset[15].replace("three", 5)
dataset[15] = dataset[15].replace("twelve", 6)
dataset[15] = dataset[15].replace("two", 7)

dataset[16] = dataset[16].replace("?", 0)

dataset[17] = dataset[17].replace("?", 0)
dataset[17] = dataset[17].replace("1bbl", 1)
dataset[17] = dataset[17].replace("2bbl", 2)
dataset[17] = dataset[17].replace("4bbl", 3)
dataset[17] = dataset[17].replace("idi", 4)
dataset[17] = dataset[17].replace("mfi", 5)
dataset[17] = dataset[17].replace("mpfi", 6)
dataset[17] = dataset[17].replace("spdi", 7)
dataset[17] = dataset[17].replace("spfi", 8)

dataset[18] = dataset[18].replace("?", 0)

dataset[19] = dataset[19].replace("?", 0)

dataset[20] = dataset[20].replace("?", 0)

dataset[21] = dataset[21].replace("?", 0)

dataset[22] = dataset[22].replace("?", 0)

dataset[23] = dataset[23].replace("?", 0)

dataset[24] = dataset[24].replace("?", 0)

dataset[25] = dataset[25].replace("?", 0)

from sklearn import preprocessing
x = preprocessing.scale(dataset)



class DimensionValueError(ValueError):
    """定义异常类"""
    pass

class PCA(object):
    """定义PCA类"""
    def __init__(self, x, n_components=None):
        """x的数据结构应为ndarray"""
        self.x = x
        self.dimension = x.shape[1]
        
        if n_components and n_components >= self.dimension:
            raise DimensionValueError("n_components error")
            
        self.n_components = n_components
        
    def cov(self):
        """求x的协方差矩阵"""
        x_T = np.transpose(self.x)                           #矩阵转秩
        x_cov = np.cov(x_T)                                  #协方差矩阵
        return x_cov
    
    def get_feature(self):
        """求协方差矩阵C的特征值和特征向量"""
        x_cov = self.cov()
        a, b = np.linalg.eig(x_cov)
        m = a.shape[0]
        c = np.hstack((a.reshape((m,1)), b))
        c_df = pd.DataFrame(c)
        c_df_sort = c_df.sort(columns=0, ascending=False)    #按照特征值大小降序排列特征向量
        return c_df_sort
        
    def explained_varience_(self):
        c_df_sort = self.get_feature()
        return c_df_sort.values[:, 0]
        
    def paint_varience_(self):
        explained_variance_ = self.explained_varience_()
        plt.figure()
        plt.plot(explained_variance_, 'k')
        plt.xlabel('n_components', fontsize=16)
        plt.ylabel('explained_variance_', fontsize=16)
        plt.show()
              
    def reduce_dimension(self):
        """指定维度降维和根据方差贡献率自动降维"""
        c_df_sort = self.get_feature()
        varience = self.explained_varience_()
        
        if self.n_components:                                #指定降维维度
            p = c_df_sort.values[0:self.n_components, 1:]
            y = np.dot(p, np.transpose(self.x))              #矩阵叉乘
            return np.transpose(y)
        
        varience_sum = sum(varience)                         #利用方差贡献度自动选择降维维度
        varience_radio = varience / varience_sum
        
        varience_contribution = 0
        for R in xrange(self.dimension):
            varience_contribution += varience_radio[R]       #前R个方差贡献度之和
            if varience_contribution >= 0.99:
                break
            
        p = c_df_sort.values[0:R+1, 1:]                      #取前R个特征向量
        y = np.dot(p, np.transpose(self.x))                  #矩阵叉乘
        return np.transpose(y)

if __name__ == '__main__':
    pca = PCA(x)
    y = pca.reduce_dimension()
        

