from method import *


data_dir = './data/'

# 将xlsx文件转化成csv文件便于处理
xlsx_to_csv(data_dir + 'Most_Recent_Cohorts_Data.xlsx', data_dir + 'Most_Recent_Cohorts_Data.csv')
xlsx_to_csv(data_dir + 'Potential_Candidate_Schools.xlsx', data_dir + 'Potential_Candidate_Schools.csv')

# 加载全部学校的统计数据,一共有7084所大学
dataFrame_ALL = pd.read_csv(data_dir + 'Most_Recent_Cohorts_Data.csv')
# 加载有潜力的学校的 unitID 信息,一共有2977所大学
dataFrame_unitID = pd.read_csv(data_dir + 'Potential_Candidate_Schools.csv')
# 从所有的学校中筛选有潜力的学校的数据 (根据筛选结果发现并不是所有的有潜力的大学都有数据,目前只有2936所大学的数据)
dataFrame_Potential = dataFrame_ALL[dataFrame_ALL['UNITID'].isin(dataFrame_unitID['UNITID'])]

# 增加一列 ACT分数
#dataFrame_Potential['ACT'] = pd.Series(np.nan, index=dataFrame_Potential.index)
# 对dataFrame增加ACT那一列的数据,并且加入其他列的有关于分数的数据
#dataFrame_Potential = dataFrame_Potential.apply(convert, axis=1)

# 数据填充值处理 PrivacySuppressed --> NaN
# 有些数据没有记录,但是并不是空值,而是一个标签'PrivacySuppressed',这导致该特征所有的数据都按照字符型来存储
# 这里使用了**kwargs方式传递参数,方便后期增加需要处理的特征
# 我们需要将标签换成NaN.然后再将数据类型转换成float64
dataFrame_Potential = dataFrame_Potential.apply(change_to_nan,k1='C150_4_POOLED_SUPP',
                                                              k2='C200_L4_POOLED_SUPP',
                                                              k3='GRAD_DEBT_MDN10YR_SUPP',
                                                              k4='RPY_3YR_RT_SUPP',
                                                              k5='gt_25k_p6',
                                                              k6='GRAD_DEBT_MDN_SUPP',
                                                              k7='md_earn_wne_p10',
                                                              k8='NPT4_PUB',
                                                              k9='NPT4_PRIV',
                                                              axis=1)
# 数据类型转换 string --> float
dataFrame_Potential = string_to_float(dataFrame_Potential,k1='C150_4_POOLED_SUPP',
                                                          k2='C200_L4_POOLED_SUPP',
                                                          k3='GRAD_DEBT_MDN10YR_SUPP',
                                                          k4='RPY_3YR_RT_SUPP',
                                                          k6='GRAD_DEBT_MDN_SUPP',
                                                          k7='md_earn_wne_p10',
                                                          k5='gt_25k_p6',
                                                          k8='NPT4_PUB',
                                                          k9='NPT4_PRIV')

# 为了补全缺失数据,试图将含义相近的数据合并在一起
# 对指定的两个特征做 C200_add_to_C150处理
dataFrame_Potential = dataFrame_Potential.apply(C200_add_to_C150, axis=1)
# 对指定的两个特征做NPT4_PRIV_add_to_NPT4_PUB处理
dataFrame_Potential = dataFrame_Potential.apply(PRIV_add_to_PUB, axis=1)
# 初步处理之后剩下的缺失数据使用平均值填充
dataFrame_Potential = repair(dataFrame_Potential)

# K-Means聚类,并且返回不同聚类数的返回评估值 :
# ss --> Calinski-Harabaz score
# sc --> Silhouette-Coefficient score
# K-Means统计测试,找出最适合的分类数
ss, sc = K_Means_test(dataFrame_Potential, 10)
# 画两种聚类评估系数的图,发现聚类分为3类更好
plot_sc(sc)
plot_ss(ss)

# 正式进行聚类,聚类数目设置为最好的类别 --> K=3
dataFrame_K, model = K_Means(dataFrame_Potential, 3)
# 因为K-Means聚类算法得出的聚类结果的顺序是随机的,这里我们经过分析发现聚类label=1的结果更好,
# 并且由于是在jupyter notebook上运行所以在没有重新聚类的前提下我们将label=1的学校保保存进csv文件里,以便于以后的分析.
# 聚类好的学校保存进csv文件里
dataFrame_K[dataFrame_K['label'] == 1].to_csv(data_dir + 'first.csv')

# 接下来是对样本进行PCA分析
# 从之前保存的文件里读取优先级第一的学校
dataFrame_K = pd.read_csv(data_dir + 'first.csv')
# 这里不知道为什么出了点小问题
# 需要转换成numpy array才可以直接传递过去
dataFrame_K['NPT4'] = np.array(dataFrame_Potential[dataFrame_Potential['UNITID'].isin(dataFrame_K['UNITID'])]['NPT4_PUB'])
# 增加一列特征 PCA PCA分析结果
dataFrame_K['PCA'] = pd.Series(np.nan, index=dataFrame_K.index)
# 增加一列特征 投资金额 Money
dataFrame_K['Money'] = pd.Series(np.nan, index=dataFrame_K.index)
# 增加一列特征 投资回报率 ROI
dataFrame_K['ROI'] = pd.Series(np.nan, index=dataFrame_K.index)
# 增加一列特征 投资时间 year
dataFrame_K['year'] = pd.Series(np.nan, index=dataFrame_K.index)

# PCA分析
dataFrame_K = pca_analyze(dataFrame_K)
# 投资金额计算 Money
dataFrame_K = Compute_Money(dataFrame_K)
# 计算投资回报率ROI
dataFrame_K = dataFrame_K.apply(ROI, axis=1)
# 对ROI进行将序排列并且取出前五十所学校,复制到dataFrame_END数据帧里面去
dataFrame_END = dataFrame_K.sort_values(by=['ROI'], ascending=False)[0:50].copy()

# 之前本来想对50所提取出来的学校再一次利用PCA分析,但是发现效果不怎么好而且多此一举,便作罢
# dataFrame_END = pca_analyze(dataFrame_END)

# 投资金额计算 Money
dataFrame_END = Compute_Money(dataFrame_END)
# 计算投资回报率 ROI
dataFrame_END  = dataFrame_END.apply(ROI, axis=1)
# 根据投资回报率排序
dataFrame_END = dataFrame_END.sort_values(by=['ROI'], ascending=False)
# 写进文件保存 dataFrame_END
save_to_excel(dataFrame_END.T, data_dir + 'end.xlsx')

# 为了便于显示,只画出前十名
dataFrame_END = dataFrame_END.sort_values(by=['ROI'], ascending=False)[0:10]
# 画出计算结果
n_groups = dataFrame_END.shape[0]
y = dataFrame_END['ROI'] * 100
index = np.arange(n_groups) + 1
bar_width = 0.6
opacity = 0.5
plt.bar(index, y, bar_width, alpha=opacity, color='g', label='ROI')
plt.xlabel('number')
plt.ylabel('Return of investment/(%)')
plt.xticks(np.linspace(1, 10, 10))
plt.title('Return of investment')
# 给柱形图加上数字标签
for a, b in zip(index, y):
    plt.text(a, b + 0.05, '%.1f' % b + '%', ha='center', va='bottom', fontsize=7.9)
plt.tight_layout()
# 保存图片
plt.savefig(data_dir + 'ROI_10.png')
plt.show()

n_groups = dataFrame_END.shape[0]
y = dataFrame_END['Money'] / 1e5
index = np.arange(n_groups) + 1
bar_width = 0.6

opacity = 0.5
plt.bar(index, y, bar_width, alpha=opacity, color='b', label='ROI')

plt.xlabel('number')
plt.ylabel('investment amount/(1x10^5$)')
plt.title('investment amount')
plt.xticks(np.linspace(1, 10, 10))
for a, b in zip(index, y):
    plt.text(a, b + 0.05, '%.1f' % b, ha='center', va='bottom', fontsize=7.9)
plt.savefig(data_dir + 'Investment_10.png')
plt.show()

# 友情提醒,论文附图里的二维图表最好不要使用python或者matlab绘制,请使用专业绘图软件,
# 同等操心程度下比代码绘制出来的不知道高到哪里去了

# 接下来是投资时间规划,因为负责建模的小姐姐写出来的公式比较复杂本菜鸟并没有能力用python符号计算库sympy算出解析解或者数值解,
# 所以用matlab来完成了最后一步,代码也在附录里
# matlab大法好!
