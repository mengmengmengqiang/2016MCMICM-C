import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numba
import time
from functools import wraps
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import PCA
from sympy import *


def xlsx_to_csv( xlsx_name, csv_name='out.csv' ):
    data_xls = pd.read_excel(xlsx_name, index_col=0)
    data_xls.to_csv(csv_name, encoding='utf-8')
    print('csv file ' + csv_name + ' saved.')


def conut_missing_rate( df ):
    na_count = df.isnull().sum().sort_values(ascending=False)
    na_rate = na_count / len(df)
    na_data = pd.concat([na_count, na_rate], axis=1, keys=['count', 'ratio'])
    return na_data


def fill_zero( df ):
    miss_cal = df.isnull().sum().sort_values(ascending=False)
    miss_cal = miss_cal[miss_cal == 10].index
    for i in miss_cal:
        if df[i][10] == 0:
            df[i] = df[i].fillna(0)

    return df


def repair_inverse( df, feature ):
    df = df.copy()
    alpha = 0.3
    n = len(df[feature[0]])
    st1 = np.zeros((1, n))
    st2 = np.zeros((1, n))
    at = np.zeros((1, n))
    bt = np.zeros((1, n))
    yhat = np.zeros((1, n))
    for f in feature:
        st1[0, 49] = df[f][49]
        st2[0, 49] = df[f][49]
        flag = True
        for a in range(1, 11):
            if flag is True:
                a = -a + 11
                # 取未缺失数据的时间序列长度
                for i in range(1, n - a):
                    # 逆时间序列预测前十年数据,所以数据要反着运算
                    i = -i - 1 + n
                    st1[0, i] = alpha * df[f][i] + (1 - alpha) * st1[0, i + 1]
                    st2[0, i] = alpha * st1[0, i] + (1 - alpha) * st2[0, i + 1]

                at = 2 * st1 - st2
                bt = alpha / (1 - alpha) * (st1 - st2)
                yhat = at + bt
                # 将预测值写入dataFrame
                # 如果预测值小于零,则写入0
                # 否则将以后的数字都写入0,退出循环
                if yhat[0, a] > 0:
                    df[f][a - 1] = yhat[0, a]
                else:
                    df[f][:a] = 0
                    flag = False
            else:  # 如果上一次预测出的数值小于零,则退出本特征值的预测
                break

    return df


def zscore( X ):
    data = X
    # 计算每一行的标准差
    data_std = np.std(data, axis=1)
    # 计算每一行的均值
    data_mean = np.mean(data, axis=1)
    # (data - data_mean)/data_std
    # 转换成矩阵直接运算
    data = np.array((np.mat(data) - np.mat(data_mean).T) / np.mat(data_std).T)
    return data


def pca_analyze( df, var ):
    """
    pca主成成分分析
    :param df: dataFrame
    :return: dataFrame
    """

    X = np.array(df[var])
    pca = PCA(n_components=0.99)
    pca.fit(X)
    # 特征向量 T
    Te = pca.components_
    # 主成分贡献率 Ratio
    Ratio = pca.explained_variance_ratio_
    # 计算标准化指标变量 zscore分析法
    X_X = zscore(X)
    # X_X * T * Ratio

    # 转换成矩阵直接运算
    Z = np.mat(X_X) * np.mat(Te).T * np.mat(Ratio).T

    # 保存进dataFrame里
    df['PCA'] = np.array(Z)

    return df


def time_series_3( df, feature=['YEAR', 'WYTCB', 'SOTCB', 'HYTCB', 'GETCB', 'BMTCB', 'NUETB', 'REPRB', 'GDPRX', 'TPOPP',
                                'TETPB', 'TETCD', 'NUETD', 'PAICD', 'TEGDS', 'TETCB', 'TEPRB'] ):
    """
    时间序列填充
        :param df: dataFrame
        :param feature: 需要时间序列预测的特征
        :return: dataFrame
        """
    df = df[feature].copy()
    # 把年份填充到 2050年
    for i in range(50, 91):
        df.loc[i] = np.nan
        df.loc[i]['YEAR'] = i + 2010 - 50
    # 时间序列预测参数设置
    alpha = 0.3
    # 观测时间序列的长度
    n = 50
    st1 = np.zeros((1, n))
    st2 = np.zeros((1, n))
    st3 = np.zeros((1, n))
    at = np.zeros((1, n))
    bt = np.zeros((1, n))
    ct = np.zeros((1, n))
    yhat = np.zeros((1, n))
    # 选取的特征
    for f in feature[1:]:
        # 创建新列
        df[f + '_predict'] = pd.Series(np.nan, index=df.index)
        # 时间序列第一个数据
        st0 = df[f][0:3].mean()
        st1[0, 0] = alpha * df[f][0] + (1 - alpha) * st0
        st2[0, 0] = alpha * st1[0, 0] + (1 - alpha) * st0
        st3[0, 0] = alpha * st2[0, 0] + (1 - alpha) * st0
        for i in range(1, 50):
            st1[0, i] = alpha * df[f][i] + (1 - alpha) * st1[0, i - 1]
            st2[0, i] = alpha * st1[0, i] + (1 - alpha) * st2[0, i - 1]
            st3[0, i] = alpha * st2[0, i] + (1 - alpha) * st3[0, i - 1]

        at = 3 * st1 - 3 * st2 + st3
        bt = 0.5 * alpha / ((1 - alpha) ** 2) * (
        (6 - 5 * alpha) * st1 - 2 * (5 - 4 * alpha) * st2 + (4 - 3 * alpha) * st3)
        ct = 0.5 * (alpha ** 2) / ((1 - alpha) ** 2) * (st1 - 2 * st2 + st3)
        yhat = at + bt + ct
        # 将预测值写入dataFrame
        df[f + '_predict'][0] = df[f][0]
        # 写入过去预测值
        df[f + '_predict'][1:50] = yhat[0, 0:49]
        # 写入未来预测值
        df[f + '_predict'][50:] = np.array(
            [at[0, 49] + bt[0, 49] * (x - 2009) + ct[0, 49] * ((x - 2009) ** 2) for x in range(2010, 2051)])

    var = feature.copy()
    for f in feature[1:]:
        var.insert(var.index(f) + 1, f + '_predict')

    return df[var]


data_dir = '/home/renqiang/Desktop/2018/energy/data/'

# 转化成csv储存,以便于处理数据
# xlsx_to_csv(data_dir + 'ProblemCData.xlsx', data_dir + 'ProblemCData.csv')
# xlsx_to_csv(data_dir + 'feature.xlsx', data_dir + 'feature.csv')

# 读取原始数据
# dataFrame_ALL = pd.read_csv(data_dir + 'ProblemCData.csv')
# 读取特征名称数据
# dataFrame_FEA = pd.read_csv(data_dir + 'feature.csv')

# 提取四个州的数据分别保存,并且去掉州的那一列特征
# 读取经过Excel处理的整张数据表
# xls = pd.ExcelFile(data_dir + 'province.xlsx')
# AZ data
# dataFrame_AZ = xls.parse('AZ')
# CA data
# dataFrame_CA = xls.parse('CA')
# NM data
# dataFrame_NM = xls.parse('NM')
# TX data
# dataFrame_TX = xls.parse('TX')
# 对于每个州都有三种缺失数据
# 对于缺失20年数据的
# TODO
# 待定

# 对于缺失17年数据的,使用拟合或者其他数据补全
# TODO

# 对于缺失十年数据的,如果第十一年数据为0,那么前十年全部补全为0
# dataFrame_AZ = fill_zero(dataFrame_AZ)
# dataFrame_CA = fill_zero(dataFrame_CA)
# dataFrame_NM = fill_zero(dataFrame_NM)
# dataFrame_TX = fill_zero(dataFrame_TX)

# 对于无法直接补0的,使用时间序列填充
# 统计缺失频率
# missing_AZ = conut_missing_rate(dataFrame_AZ)
# missing_CA = conut_missing_rate(dataFrame_CA)
# missing_NM = conut_missing_rate(dataFrame_NM)
# missing_TX = conut_missing_rate(dataFrame_TX)

# 运行时间过长,不必要再运行一次
# 补全缺失值
# dataFrame_AZ = repair_inverse(dataFrame_AZ, missing_AZ[missing_AZ['count'] == 10].index)
# dataFrame_CA = repair_inverse(dataFrame_CA, missing_CA[missing_CA['count'] == 10].index)
# dataFrame_NM = repair_inverse(dataFrame_NM, missing_NM[missing_NM['count'] == 10].index)
# dataFrame_TX = repair_inverse(dataFrame_TX, missing_TX[missing_TX['count'] == 10].index)

# 保存补全过的数据到excel文件里
# excelWriter=pd.ExcelWriter(data_dir + 'repaired_data.xlsx', engine='openpyxl')
# dataFrame_AZ.to_excel(excel_writer=excelWriter, sheet_name="AZ",index=None)
# dataFrame_CA.to_excel(excel_writer=excelWriter, sheet_name="CA",index=None)
# dataFrame_NM.to_excel(excel_writer=excelWriter, sheet_name="NM",index=None)
# dataFrame_TX.to_excel(excel_writer=excelWriter, sheet_name="TX",index=None)
# excelWriter.save()
# excelWriter.close()


# 读取经过Excel处理的整张数据表
xls = pd.ExcelFile(data_dir + 'repaired_data.xlsx')
# AZ data
dataFrame_AZ = xls.parse('AZ')
# CA data
dataFrame_CA = xls.parse('CA')
# NM data
dataFrame_NM = xls.parse('NM')
# TX data
dataFrame_TX = xls.parse('TX')
# 统计缺失率
# missing_AZ = conut_missing_rate(dataFrame_AZ)
# missing_CA = conut_missing_rate(dataFrame_CA)
# missing_NM = conut_missing_rate(dataFrame_NM)
# missing_TX = conut_missing_rate(dataFrame_TX)

# 取一行
# dataFrame_NM.loc[49][1:]
# 创建新dataFrame, 包含四个州的2009年数据
# dataFrame_PCA = pd.DataFrame(np.nan,index=['AZ','CA','NM','TX'],columns=dataFrame_AZ.columns)
# dataFrame_PCA.loc['AZ'] = dataFrame_AZ.loc[49]
# dataFrame_PCA.loc['CA'] = dataFrame_CA.loc[49]
# dataFrame_PCA.loc['NM'] = dataFrame_NM.loc[49]
# dataFrame_PCA.loc['TX'] = dataFrame_TX.loc[49]
# 新加一行PCA得分
# dataFrame_PCA['PCA'] = pd.Series(np.nan, index=dataFrame_PCA.index)

# 主成成分分析
# dataFrame_PCA = pca_analyze(dataFrame_PCA)
# 主成成分按照降序排列打印出来
# dataFrame_PCA['PCA'].sort_values(ascending=False)

# 读取需要预测的四个州数据
# dataFrame_AZ = pd.read_excel(data_dir + 'predict/AZ.xlsx')
# dataFrame_CA = pd.read_excel(data_dir + 'predict/CA.xlsx')
# dataFrame_NM = pd.read_excel(data_dir + 'predict/NM.xlsx')
# dataFrame_TX = pd.read_excel(data_dir + 'predict/TX.xlsx')

# 三次指数平滑时间序列
# dataFrame_AZ_3 = time_series_3(dataFrame_AZ)
# dataFrame_CA_3 = time_series_3(dataFrame_CA)
# dataFrame_NM_3 = time_series_3(dataFrame_NM)
# dataFrame_TX_3 = time_series_3(dataFrame_TX)

# 保存补全过的数据到excel文件里
# excelWriter=pd.ExcelWriter(data_dir + 'time_series_3_V2.0.xlsx', engine='openpyxl')
# dataFrame_AZ_3.to_excel(excel_writer=excelWriter, sheet_name="AZ",index=None)
# dataFrame_CA_3.to_excel(excel_writer=excelWriter, sheet_name="CA",index=None)
# dataFrame_NM_3.to_excel(excel_writer=excelWriter, sheet_name="NM",index=None)
# dataFrame_TX_3.to_excel(excel_writer=excelWriter, sheet_name="TX",index=None)
# excelWriter.save()
# excelWriter.close()

# 读取对应方式补全数据的整张数据表
# xls = pd.ExcelFile(data_dir + 'predict/prediction_data.xlsx')
# AZ data
# dataFrame_AZ = xls.parse('AZ')
# CA data
# dataFrame_CA = xls.parse('CA')
# NM data
# dataFrame_NM = xls.parse('NM')
# TX data
# dataFrame_TX = xls.parse('TX')

# 创建新dataFrame, 包含四个州的2025年数据
# dataFrame_PCA_25 = pd.DataFrame(np.nan,index=['AZ','CA','NM','TX'],columns=dataFrame_AZ.columns)
# dataFrame_PCA_25.loc['AZ'] = dataFrame_AZ.loc[65]
# dataFrame_PCA_25.loc['CA'] = dataFrame_CA.loc[65]
# dataFrame_PCA_25.loc['NM'] = dataFrame_NM.loc[65]
# dataFrame_PCA_25.loc['TX'] = dataFrame_TX.loc[65]

# 2050主成成分分析
# dataFrame_PCA_25 = pca_analyze(dataFrame_PCA_25)

# 创建新dataFrame, 包含四个州的2050年数据
# dataFrame_PCA_50 = pd.DataFrame(np.nan,index=['AZ','CA','NM','TX'],columns=dataFrame_AZ.columns)
# dataFrame_PCA_50.loc['AZ'] = dataFrame_AZ.loc[90]
# dataFrame_PCA_50.loc['CA'] = dataFrame_CA.loc[90]
# dataFrame_PCA_50.loc['NM'] = dataFrame_NM.loc[90]
# dataFrame_PCA_50.loc['TX'] = dataFrame_TX.loc[90]

# 主成成分分析
# dataFrame_PCA_50 = pca_analyze(dataFrame_PCA_50)
# 主成成分按照降序排列打印出来
# dataFrame_PCA_50['PCA'].sort_values(ascending=False)

## 敏感度分析的主成成分分析
# 创建新dataFrame, 包含四个州的2009年数据
# dataFrame_PCA = pd.DataFrame(np.nan,index=['AZ','CA','NM','TX'],columns=dataFrame_AZ.columns)
# dataFrame_PCA.loc['AZ'] = dataFrame_AZ.loc[49]
# dataFrame_PCA.loc['CA'] = dataFrame_CA.loc[49]
# dataFrame_PCA.loc['NM'] = dataFrame_NM.loc[49]
# dataFrame_PCA.loc['TX'] = dataFrame_TX.loc[49]

# 主成成分分析
# 'WYTCB', 'SOTCB', 'HYTCB', 'GETCB', 'BMTCB', 'NUETB', 'REPRB', 'GDPRX', 'TPOPP', 'TETPB', 'TETCD', 'NUETD'
# dataFrame_PCA = pca_analyze(dataFrame_PCA, ['TEGDS', 'PAICD', 'ELEXD', 'TETCD', 'TETPB'])
# dataFrame_PCA = pca_analyze(dataFrame_PCA, ['TEGDS', 'PAICD', 'ELEXD', 'WYTCB', 'SOTCB', 'HYTCB', 'GETCB', 'BMTCB', 'NUETB', 'REPRB', 'GDPRX', 'TPOPP', 'TETPB', 'TETCD', 'NUETD'])
# 主成成分按照降序排列打印出来
# dataFrame_PCA['PCA'].sort_values(ascending=False)

# print('OK!')