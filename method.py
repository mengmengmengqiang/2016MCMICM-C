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
import sympy as sy


def xlsx_to_csv( xlsx_name, csv_name='out.csv' ):
    """
    把xlsx文件转换成csv文件
        :param xlsx_name: 需要转换的源文件 带路径
        :param csv_name: 转换成的目标文件名 带路径
        :return: 
        """
    data_xls = pd.read_excel(xlsx_name, index_col=0)
    data_xls.to_csv(csv_name, encoding='utf-8')
    print('csv file ' + csv_name + ' saved.')


def act( old_sat ):
    """

    :param old_sat: old SAT score
    :return: ACT score
    """
    result = 7.590765625e-14 * old_sat ** 4 \
             - 1.422768675e-9 * old_sat ** 3 \
             + 5.844760976e-6 * old_sat ** 2 \
             + 0.007704713238 * old_sat \
             + 1.423737361
    if result > 36:
        result = 36
    return result


def timethis( func ):
    """
    测量函数运行占用的时间
    :param func: 函数名
    :return: 函数运行时间
    """

    @wraps(func)
    def wrapper( *args, **kwargs ):
        start = time.perf_counter()
        r = func(*args, **kwargs)
        end = time.perf_counter()
        print('{}.{} : {}'.format(func.__module__, func.__name__, end - start))
        return r

    return wrapper


def anova( frame, qualitative ):
    """
    一元方差分析
    :param frame: 数据集合
    :param qualitative:  变量类型结合
    :return: 方差集合
    """
    anv = pd.DataFrame()
    anv['feature'] = qualitative
    pvals = []
    for c in qualitative:
        samples = []
        for cls in frame[c].unique():
            s = frame[frame[c] == cls]['SalePrice'].values
            samples.append(s)  # 某特征下不通取值对应的房价组合形成二维列表
        pval = stats.f_oneway(*samples)[1]  # 一元方差分析得到 F, P,要的是 P, P越小,对方差的影响越大.
        pvals.append(pval)
    anv['pval'] = pvals
    return anv.sort_values('pval')


@numba.jit()
def encode( frame, feature ):
    """
    对所有类型变量,依照各个类型变量的不同取值对应的样本集合内房价的均值,按照房价均值高低
    对此变量当前取值确定其相对数值1,2,3,4等等,相当于对类型变量复制使其成为连续变量.
    对此方法采用了与One-Hot编码不同的方法来处理离散数据,值得学习
    注意:此函数会直接在原frame的DataFrame内创建新的一列来存放feature编码后的值.
    :param frame: 数据集合
    :param feature: 特性
    """
    ordering = pd.DataFrame()
    ordering['val'] = frame[feature].unique()
    ordering.index = ordering.val
    # 以下 groupby()操作可以将某一feature下同一取值的数据整个到一起，结合mean()可以直接得到该特征不同取值的房价均值
    ordering['price_mean'] = frame[[feature, 'SalePrice']].groupby(feature).mean()['SalePrice']
    ordering = ordering.sort_values('price_mean')
    ordering['order'] = range(1, ordering.shape[0] + 1)
    ordering = ordering['order'].to_dict()
    for attr_v, score in ordering.items():
        # e.g. qualitative[2]:{'Grvl':1, 'MISSING':3,'Pave':2}
        frame.loc[frame[feature] == attr_v, feature + '_E'] = score


def spearman( frame, features ):
    """
    采用"斯皮尔曼等级相关"来计算变量与房价的相关特性
    可对encode处理后的等级变量以及其他与房价的相关性进行更好的评价(特别是对于非线性关系)

    :param frame: 数据集
    :param features: 特性
    """

    spr = pd.DataFrame()
    spr['feature'] = features
    spr['corr'] = [frame[f].corr(frame['SalePrice'], 'spearman') for f in features]
    spr = spr.sort_values('corr')
    plt.figure(figsize=(6, 0.25 * len(features)))
    sns.barplot(data=spr, y='feature', x='corr', orient='h')
    plt.show()  # 显示图像


def convert( df ):
    global score
    if (np.isnan(df['SAT_AVG']) & np.isnan(df['SAT_AVG_ALL']) & np.isnan(df['ACTCMMID'])):
        df['ACT'] = np.nan
    else:
        if not np.isnan(df['SAT_AVG']):
            score = df['SAT_AVG']
        elif not np.isnan(df['SAT_AVG_ALL']):
            score = df['SAT_AVG_ALL']
        elif not np.isnan(df['ACTCMMID']):
            score = df['ACTCMMID']
        df['ACT'] = score
    return df


def save_to_excel( data, path_to_file ):
    writer = pd.ExcelWriter(path_to_file)
    df1 = pd.DataFrame(data=data.T)
    df1.to_excel(writer, 'Sheet1')
    writer.save()


def change_to_nan( df, **kwargs ):
    """
        对 kwargs 里面所有的
        特征缺失值PrivacySuppressed转换成 NaN
    :param df: 
    :return: 
    """
    for feature in kwargs:
        if df[kwargs[feature]] == 'PrivacySuppressed':
            df[kwargs[feature]] = np.nan
            #    if df['C200_L4_POOLED_SUPP'] == 'PrivacySuppressed':
            #        df['C200_L4_POOLED_SUPP'] = np.nan
            #    if df['GRAD_DEBT_MDN10YR_SUPP'] == 'PrivacySuppressed':
            #        df['GRAD_DEBT_MDN10YR_SUPP'] = np.nan
    return df


def C200_add_to_C150( df ):
    if np.isnan(df['C150_4_POOLED_SUPP']) & (not np.isnan(df['C200_L4_POOLED_SUPP'])):
        df['C150_4_POOLED_SUPP'] = df['C200_L4_POOLED_SUPP']

    return df


def string_to_float( df, **kwargs ):
    """
    把对应特征值的一列字符型的变量转换成浮点型
    :param df: dataFrame
    :param kwargs: 特征值的key
    :return: dataFrame
    """
    for feature in kwargs:
        df[kwargs[feature]] = df[kwargs[feature]].astype(float)
    return df


def repair( source_df ):
    """
        补全缺失数据,数值型变量使用平均值补全,非数值型变量使用 MISSING 补全
                :param df: dataFrame
                :return: dataFrame
                """
    df = source_df.copy()  # 在源dataFrame的备份里操作
    quantity = [attr for attr in df.columns if df.dtypes[attr] != 'object']  # 数值变量集合
    quality = [attr for attr in df.columns if df.dtypes[attr] == 'object']  # 非数值变量集合
    for c in quality:  # 非数值变量缺失值补全
        df[c] = df[c].astype('category')
        if df[c].isnull().any():
            df[c] = df[c].cat.add_categories(['MISSING'])
            df[c] = df[c].fillna('MISSING')

    # 连续变量缺失值补全
    quantity_miss_cal = df[quantity].isnull().sum().sort_values(ascending=False)
    missing_cols = quantity_miss_cal[quantity_miss_cal > 0].index  # 数据缺失率大于0的索引
    df[missing_cols] = df[missing_cols].fillna(df.mean()[missing_cols])  # 缺失数据以平均值补全
    # df[missing_cols].isnull().sum()  # 验证缺失值是否都已补全

    return df


def conut_missing_rate( df ):
    """
            补全缺失数据,数值型变量使用平均值补全,非数值型变量使用 MISSING 补全
            对原dataFrame做一份拷贝,在拷贝里修改,修改好了之后再返回修改的拷贝dataFrame
                    :param source_df: dataFrame
                    :return: dataFrame
                    """
    na_count = df.isnull().sum().sort_values(ascending=False)
    na_rate = na_count / len(df)
    na_data = pd.concat([na_count, na_rate], axis=1, keys=['count', 'ratio'])
    return na_data


def K_Means_test( df, K ):
    """
            k-Means聚类,并且计算评估值
                            :param df: dataFrame
                            :return:  ss, sc 分别是 Calinski-Harabaz score 和 Silhouette-Coefficient score      
                            """
    # 创建需要索引的特征
    var = ['UNITID', 'PCTPELL', 'PCTFLOAN', 'gt_25k_p6', 'md_earn_wne_p10', 'RPY_3YR_RT_SUPP', 'C150_4_POOLED_SUPP',
           'GRAD_DEBT_MDN_SUPP']
    # 读取需要聚类的数据以及UNITID
    dataFrame_K = df[var].copy()

    # K-Means聚类
    # 设置要聚类的字段
    loan = np.array(dataFrame_K[var[1:]])
    # 使用Calinski-Harabaz Index评估方法进行分群效果评估
    ss = []
    # 使用轮廓系数Silhouette-Coefficient 评估方法进行效果评估
    sc = []
    for classify in range(2, K):
        # 设置类别为classify,将数据带入到聚类模型中
        kmeans_model = KMeans(n_clusters=classify).fit(loan)
        # 查看聚类结果
        # kmeans_model.cluster_centers_
        # 测试聚类结果
        # TUDO
        # 在原始数据表格中添加聚类结果标签
        # dataFrame_K['label'] = kmeans_model.labels_
        labels = kmeans_model.labels_
        ss.append(metrics.calinski_harabaz_score(loan, labels))
        sc.append(metrics.silhouette_score(loan, labels, metric='euclidean'))

    return ss, sc


def plot_ss( ss ):
    """
    画 Calinski-Harabaz score
                    :param ss: 要画的Calinski-Harabaz score
                    :return: None              
                    """
    # %matplotlib inline
    plt.plot(range(2, len(ss) + 2), ss, 'o-.')
    plt.xlabel('value of k')
    plt.ylabel('Calinski-Harabaz score')
    # plt.text(r'Calinski-Harabaz Index')
    plt.grid(False)
    plt.savefig('/home/renqiang/Desktop/2018MCMICM/TheThirdPractice/goodgrant/picture/Calinski-Harabaz.png')
    plt.show()
    return None


def plot_sc( sc ):
    """
        画 Silhouette-Coefficient score
                        :param ss: 要画的Silhouette-Coefficient score
                        :return: None              
                        """
    # %matplotlib inline
    plt.plot(range(2, len(sc) + 2), sc, 'o-.')
    plt.xlabel('value of k')
    plt.ylabel('Silhouette-Coefficient score')
    # plt.text(r'Silhouette-Coefficient Index')
    plt.grid(False)
    plt.savefig('/home/renqiang/Desktop/2018MCMICM/TheThirdPractice/goodgrant/picture/Silhouette-Coefficient.png')
    plt.show()
    return None


def K_Means( df, K ):
    """
            k-Means聚类
                            :param df: dataFrame
                            :return:  添加聚类标签的dataFrame_K, 聚类后的model
                            """
    # 创建需要索引的特征
    var = ['UNITID', 'PCTPELL', 'PCTFLOAN', 'gt_25k_p6', 'md_earn_wne_p10', 'RPY_3YR_RT_SUPP', 'C150_4_POOLED_SUPP',
           'GRAD_DEBT_MDN_SUPP']
    # 读取需要聚类的数据以及UNITID
    dataFrame_K = df[var].copy()

    # K-Means聚类
    # 设置要聚类的字段
    loan = np.array(dataFrame_K[var[1:]])
    # 设置类别数目为K,将数据带入到聚类模型中
    kmeans_model = KMeans(n_clusters=K).fit(loan)
    # 在原始数据表格中添加聚类结果标签
    dataFrame_K['label'] = kmeans_model.labels_

    return dataFrame_K, kmeans_model


def zscore( X ):
    """
    z-score标准化
    :param X: numpy 数组
    :return: numpy 数组
    """
    data = X
    # 计算每一行的标准差
    data_std = np.std(data, axis=1)
    # 计算每一行的均值
    data_mean = np.mean(data, axis=1)
    # (data - data_mean)/data_std
    # 转换成矩阵直接运算
    data = np.array((np.mat(data) - np.mat(data_mean).T)/np.mat(data_std).T)
    return data


def PRIV_add_to_PUB( df ):
    """
    合并特征值NPT4_PUB NPT4_PRIV
    :param df: dataFrame
    :return: dataFrame
    """
    if np.isnan(df['NPT4_PUB']) & (not np.isnan(df['NPT4_PRIV'])):
        df['NPT4_PUB'] = df['NPT4_PRIV']

    return df


def ROI( df ):
    """
    计算投资回报率
    :param df: dataFrame 
    :return: dataFrame
    """
    df['ROI'] = (10 * df['md_earn_wne_p10'] * df['C150_4_POOLED_SUPP'] - 4 * df['NPT4'] - df['GRAD_DEBT_MDN_SUPP']) / \
                df['Money']
    return df


def get_t( df ):
    """
    计算投资时间
    :param df: dataFrame 
    :return: dataFrame
    """
    t = sy.Symbol('t')
    df['year'] = sy.solve(df['Money'] / (
    10 * 1.0065 ** t * df['C150_4_POOLED_SUPP'] * (df['md_earn_wne_p10'] + 16000 * sy.log(t, 10)) - df[
        'GRAD_DEBT_MDN_SUPP'] - 4 * df['NPT4']) - t, t)

    return df


def pca_analyze( df ):
    """
    pca主成成分分析
    :param df: dataFrame
    :return: dataFrame
    """
    var = ['UNITID', 'PCTPELL', 'PCTFLOAN', 'gt_25k_p6', 'md_earn_wne_p10', 'RPY_3YR_RT_SUPP', 'C150_4_POOLED_SUPP',
           'GRAD_DEBT_MDN_SUPP']
    X = np.array(df[var[1:]])
    pca = PCA(n_components=0.99)
    pca.fit(X)
    # 特征向量 T
    Te = pca.components_
    # 主成分贡献率 Ratio
    Ratio = pca.explained_variance_ratio_
    # 计算标准化指标变量 zscore分析法
    X_X = zscore(X)
    # X_X * Te * Ratio

    # 转换成矩阵直接运算
    Z = np.mat(X_X) * np.mat(Te).T * np.mat(Ratio).T

    # 保存进dataFrame里
    df['PCA'] = np.array(Z)

    return df


def Compute_Money( df ):
    """
    计算分配得到的投资金额
    :param df: dataFrame
    :return: dataFrame
    """
    df['Money'] = df['PCA'] / df['PCA'].sum() * 1e8
    return df