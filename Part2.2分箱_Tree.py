import pandas as pd
import numpy as np

#读取数据
trainData = pd.read_csv('test1.csv')
allFeatures = list(trainData.columns)
allFeatures.remove('Unnamed: 0')

## 同样需要将数据分成数值型变量和非数值型变量，并且对非数值型变量进行合并，编码等操作
## 这里只选取数据集中某个变量进行测验

def calc_score_median(sample_set, var):
    '''
    计算相邻评分的中位数，以便进行决策树二元切分
    param sample_set: 待切分样本
    param var: 分割变量名称
    '''
    var_list = list(np.unique(sample_set[var]))
    var_median_list = []
    for i in range(len(var_list) -1):
        var_median = (var_list[i] + var_list[i+1]) / 2
        var_median_list.append(var_median)
    return var_median_list

def choose_best_split(sample_set, var, min_sample):
    '''
    使用CART分类决策树选择最好的样本切分点
    返回切分点
    param sample_set: 待切分样本
    param var: 分割变量名称
    param min_sample: 待切分样本的最小样本量(限制条件)
    '''
    # 根据样本评分计算相邻不同分数的中间值
    score_median_list = calc_score_median(sample_set, var)
    median_len = len(score_median_list)
    sample_cnt = sample_set.shape[0]
    sample1_cnt = sum(sample_set['result'])
    sample0_cnt =sample_cnt- sample1_cnt
    Gini = 1 - np.square(sample1_cnt / sample_cnt) - np.square(sample0_cnt / sample_cnt)

    bestGini = 0.0; bestSplit_point = 0.0; bestSplit_position = 0.0
    for i in range(median_len):
        left = sample_set[sample_set[var] < score_median_list[i]]
        right = sample_set[sample_set[var] > score_median_list[i]]

        left_cnt = left.shape[0]
        right_cnt = right.shape[0]
        left1_cnt = sum(left['result'])
        right1_cnt = sum(right['result'])
        left0_cnt = left_cnt - left1_cnt
        right0_cnt =right_cnt-right1_cnt
        left_ratio = left_cnt / sample_cnt
        right_ratio = right_cnt / sample_cnt

        if left_cnt < min_sample or right_cnt < min_sample:
            continue

        Gini_left = 1 - np.square(left1_cnt / left_cnt) - np.square(left0_cnt / left_cnt)
        Gini_right = 1 - np.square(right1_cnt / right_cnt) - np.square(right0_cnt / right_cnt)
        Gini_temp = Gini - (left_ratio * Gini_left + right_ratio * Gini_right)
        if Gini_temp > bestGini:
            bestGini = Gini_temp
            bestSplit_point = score_median_list[i]
            if median_len > 1:
                bestSplit_position = i / (median_len - 1)
            else:
                bestSplit_position = i / median_len
        else:
            continue

    Gini = Gini - bestGini
    return bestSplit_point, bestSplit_position

def bining_data_split(sample_set, var, min_sample, split_list):
    '''
    划分数据找到最优分割点list
    param sample_set: 待切分样本
    param var: 分割变量名称
    param min_sample: 待切分样本的最小样本量(限制条件)
    param split_list: 最优分割点list
    '''
    split, position = choose_best_split(sample_set, var, min_sample)
    if split != 0.0:
        split_list.append(split)
    # 根据分割点划分数据集，继续进行划分
    sample_set_left = sample_set[sample_set[var] < split]
    sample_set_right = sample_set[sample_set[var] > split]
    # 如果左子树样本量超过2倍最小样本量，且分割点不是第一个分割点，则切分左子树
    if len(sample_set_left) >= min_sample * 2 and position not in [0.0, 1.0]:
        bining_data_split(sample_set_left, var, min_sample, split_list)
    else:
        None
    # 如果右子树样本量超过2倍最小样本量，且分割点不是最后一个分割点，则切分右子树
    if len(sample_set_right) >= min_sample * 2 and position not in [0.0, 1.0]:
        bining_data_split(sample_set_right, var, min_sample, split_list)
    else:
        None

def get_bestsplit_list(sample_set, var):
    '''
    根据分箱得到最优分割点list
    param sample_set: 待切分样本
    param var: 分割变量名称
    '''
    # 计算最小样本阈值（终止条件）
    min_df = sample_set.shape[0] * 0.05
    split_list = []
    # 计算第一个和最后一个分割点
    bining_data_split(sample_set, var, min_df, split_list)
    return split_list


print(get_bestsplit_list(trainData, 'V01'))