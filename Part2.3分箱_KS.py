import pandas as pd
import numpy as np

#读取数据
trainData = pd.read_csv('test1.csv')
allFeatures = list(trainData.columns)
allFeatures.remove('Unnamed: 0')
datalines = trainData.shape[0]

## 同样需要将数据分成数值型变量和非数值型变量，并且对非数值型变量进行合并，编码等操作
## 这里只选取数据集中某个变量进行测验

def ks_bin(data_, limit):
    # 好坏的个数
    g = data_.ix[:, 1].value_counts()[1]
    b = data_.ix[:, 1].value_counts()[0]
    # 每个段不同的好坏数量
    data_cro = pd.crosstab(data_.ix[:, 0], data_.ix[:, 1])
    # 计算不同段的好坏百分比
    data_cro[0] = data_cro[0] / g
    data_cro[1] = data_cro[1] / b
    # 数值进行累加
    data_cro_cum = data_cro.cumsum()
    # 计算不同段的好坏比差别
    ks_list = abs(data_cro_cum[1] - data_cro_cum[0])
    # 按照最大差别进行排序并且变成列表
    ks_list_index = ks_list.nlargest(len(ks_list)).index.tolist()
    for i in ks_list_index:
        data_1 = data_[data_.ix[:, 0] <= i]
        data_2 = data_[data_.ix[:, 0] > i]
        if len(data_1) >= limit and len(data_2) >= limit:
            break
    return i

def ks_zone(data_, list_):
    list_zone = list()
    list_.sort()
    n = 0
    for i in list_:
        m = sum(data_.ix[:, 0] <= i) - n
        n = sum(data_.ix[:, 0] <= i)
        list_zone.append(m)
    list_zone.append(datalines - sum(list_zone))
    max_index = list_zone.index(max(list_zone))
    if max_index == 0:
        rst = [data_.ix[:, 0].unique().min(), list_[0]]
    elif max_index == len(list_):
        # 选取项 最大项
        rst = [list_[-1], data_.ix[:, 0].unique().max()]
    else:
        rst = [list_[max_index - 1], list_[max_index]]
    return rst

def best_ks_box(data,var_name,box_num):
    # 单独取某一类的数据
    data = data[[var_name, 'result']]
    data_ = data.copy()
    limit_ = data.shape[0] / 20  # 总体的5%
    """"
    循环体
    """
    zone = list()
    for i in range(box_num - 1):
        print(i)
        ks_ = ks_bin(data_, limit_)
        zone.append(ks_)
        new_zone = ks_zone(data, zone)
        data_ = data[(data.ix[:, 0] > new_zone[0]) & (data.ix[:, 0] <= new_zone[1])]

    """
    构造分箱明细表
    """
    zone.append(data.ix[:, 0].unique().max())
    zone.append(data.ix[:, 0].unique().min())
    zone.sort()
    df_ = pd.DataFrame(columns=[0, 1])
    for i in range(len(zone) - 1):
        if i == 0:
            data_ = data[(data.ix[:, 0] >= zone[i]) & (data.ix[:, 0] <= zone[i + 1])]
        else:
            data_ = data[(data.ix[:, 0] > zone[i]) & (data.ix[:, 0] <= zone[i + 1])]
        data_cro = pd.crosstab(data_.ix[:, 0], data_.ix[:, 1])
        df_.loc['{0}-{1}'.format(data_cro.index.min(), data_cro.index.max())] = data_cro.apply(sum)
    return df_


print(best_ks_box(trainData,'V01',5))
