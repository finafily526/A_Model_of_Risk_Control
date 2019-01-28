import pandas as pd
import numpy as np
import numbers
import random
import statsmodels.api as sm
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def MissingCategorial(df,x):
    missing_vals = df[x].map(lambda x: int(x!=x))
    return sum(missing_vals)*1.0/df.shape[0]

def MissingContinuous(df,x):
    missing_vals = df[x].map(lambda x: int(np.isnan(x)))
    return sum(missing_vals) * 1.0 / df.shape[0]


# 读取数据
allData = pd.read_csv('test.csv')
# 提取对应的字段
allFeatures = list(allData.columns)
allFeatures.remove('Unnamed: 0')

# 对应的将字段区分为字符型变量和数字型变量
numerical_var = []
for col in allFeatures:
    # 删除只有一个值的变量
    if len(set(allData[col])) == 1:
        print('delete {} from the dataset because it is a constant'.format(col))
        del allData[col]
        allFeatures.remove(col)
    else:
        # 种类大于5 并且种类为数字，连续型变量加入这个
        uniq_valid_vals = [i for i in allData[col] if i == i]
        uniq_valid_vals = list(set(uniq_valid_vals))
        if len(uniq_valid_vals) >= 10 and isinstance(uniq_valid_vals[0], numbers.Real):
            numerical_var.append(col)
# 字符变量是非连续型变量
categorical_var = [i for i in allFeatures if i not in numerical_var]

#检查变量的最多值的占比情况,以及每个变量中占比最大的值
records_count = allData.shape[0]
col_most_values,col_large_value = {},{}
for col in allFeatures:
    value_count = allData[col].groupby(allData[col]).count()
    col_most_values[col] = max(value_count)/records_count
    large_value = value_count[value_count== max(value_count)].index[0]
    col_large_value[col] = large_value
col_most_values_df = pd.DataFrame.from_dict(col_most_values, orient = 'index')
col_most_values_df.columns = ['max percent']
col_most_values_df = col_most_values_df.sort_values(by = 'max percent', ascending = False)
pcnt = list(col_most_values_df['max percent'])

plt.bar(range(len(pcnt)), height = pcnt)
plt.title('Largest Percentage of Single Value in Each Variable')
plt.show()

#计算多数值占比超过90%的字段中，少数值的坏样本率是否会显著高于多数值,如果显著高则需要保留
large_percent_cols = list(col_most_values_df[col_most_values_df['max percent']>=0.9].index)
bad_rate_diff = {}
for col in large_percent_cols:
    large_value = col_large_value[col]
    temp = allData[[col,'result']]
    # 最多的值为1 ，其他为0
    temp[col] = temp.apply(lambda x: int(x[col]==large_value),axis=1)
    bad_rate = temp.groupby(col).mean()
    # 如果其他项坏账率为0
    if bad_rate.iloc[0]['result'] == 0:
        bad_rate_diff[col] = 0
        continue
    # 少数样本bad rate /多数样本 bad rate 如果log结果为负，说明多数样本的bad rate 大
    bad_rate_diff[col] = np.log(bad_rate.iloc[0]['result']/bad_rate.iloc[1]['result'])
bad_rate_diff_sorted = sorted(bad_rate_diff.items(),key=lambda x: x[1], reverse=True)
bad_rate_diff_sorted_values = [x[1] for x in bad_rate_diff_sorted]
plt.bar(x = range(len(bad_rate_diff_sorted_values)), height = bad_rate_diff_sorted_values)
plt.show()

#由于所有的少数值的坏样本率并没有显著高于多数值，意味着这些变量可以直接剔除
#（或者剔除掉上面为负值的情况，说明多数数据结果不好）
## 剔除为负的值

for col in bad_rate_diff_sorted:
    if col in numerical_var:
        numerical_var.remove(col)
    else:
        categorical_var.remove(col)
    del allData[col]


'''
对类别型变量，如果缺失超过80%, 就删除，否则当成特殊的状态,由于之前已经对所有值进行填充，这里下面没有什么用
'''
missing_pcnt_threshould_1 = 0.8
for col in categorical_var:
    missingRate = MissingCategorial(allData,col)
    print('{0} has missing rate as {1}'.format(col,missingRate))
    if missingRate > missing_pcnt_threshould_1:
        print('missssssssss!')
        categorical_var.remove(col)
        del allData[col]
    # 对于数字则添加缺失值-9999
    if 0 < missingRate < missing_pcnt_threshould_1:
        uniq_valid_vals = [i for i in allData[col] if i == i]
        uniq_valid_vals = list(set(uniq_valid_vals))
        if isinstance(uniq_valid_vals[0], numbers.Real):
            missing_position = allData.loc[allData[col] != allData[col]][col].index
            not_missing_sample = [-9999]*len(missing_position)
            allData.loc[missing_position, col] = not_missing_sample
        else:
            # In this way we convert NaN to NAN, which is a string instead of np.nan
            allData[col] = allData[col].map(lambda x: str(x).upper())

allData_bk = allData.copy()

'''
检查数值型变量
'''
missing_pcnt_threshould_2 = 0.8
deleted_var = []
for col in numerical_var:
    #print(col)
    missingRate = MissingContinuous(allData, col)
    print('{0} has missing rate as {1}'.format(col, missingRate))
    if missingRate > missing_pcnt_threshould_2:
        print('misssssssss!')
        deleted_var.append(col)
        print('we delete variable {} because of its high missing rate'.format(col))
    else:
        if missingRate > 0:
            not_missing = allData.loc[allData[col] == allData[col]][col]
            #makeuped = allData[col].map(lambda x: MakeupRandom(x, list(not_missing)))
            missing_position = allData.loc[allData[col] != allData[col]][col].index
            not_missing_sample = random.sample(list(not_missing), len(missing_position))
            allData.loc[missing_position,col] = not_missing_sample
            #del allData[col]
            #allData[col] = makeuped
            missingRate2 = MissingContinuous(allData, col)
            print('missing rate after making up is:{}'.format(str(missingRate2)))
# 如果存在需要删除的变量 则进行删除
if deleted_var != []:
    for col in deleted_var:
        numerical_var.remove(col)
        del allData[col]

allData.to_csv('test1.csv', header=True, index=False)
