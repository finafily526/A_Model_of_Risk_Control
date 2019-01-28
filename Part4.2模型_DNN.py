import pandas as pd
import numpy as np
import numbers
import pickle
import operator
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import SKCompat

def Outlier_Dectection(df,x):
    '''
    :param df:
    :param x:
    :return:
    '''
    p25, p75 = np.percentile(df[x], 25),np.percentile(df[x], 75)
    d = p75 - p25
    upper, lower =  p75 + 1.5*d, p25-1.5*d
    truncation = df[x].map(lambda x: max(min(upper, x), lower))
    return truncation

#读取数据
allData = pd.read_csv('test1.csv')
##只计算统计变量
allFeatures = list(allData.columns)
allFeatures.remove('Unnamed: 0')
allFeatures.remove('result')


#检查是否有常数型变量，并且检查是类别型还是数值型变量
numerical_var = []
for col in allFeatures:
    if len(set(allData[col])) == 1:
        print('delete {} from the dataset because it is a constant'.format(col))
        del allData[col]
        allFeatures.remove(col)
    else:
        uniq_valid_vals = [i for i in allData[col] if i == i]
        uniq_valid_vals = list(set(uniq_valid_vals))
        if len(uniq_valid_vals) >= 10 and isinstance(uniq_valid_vals[0], numbers.Real):
            numerical_var.append(col)

categorical_var = [i for i in allFeatures if i not in numerical_var]

'''
用one-hot对类别型变量进行编码
'''
dummy_map = {}
dummy_columns = []
for raw_col in categorical_var:
    dummies = pd.get_dummies(allData.loc[:, raw_col], prefix=raw_col)
    col_onehot = pd.concat([allData[raw_col], dummies], axis=1)
    col_onehot = col_onehot.drop_duplicates()
    allData = pd.concat([allData, dummies], axis=1)
    del allData[raw_col]
    dummy_map[raw_col] = col_onehot
    dummy_columns = dummy_columns + list(dummies)

'''
对极端值变量做处理。
'''
max_min_standardized = {}
for col in numerical_var:
    truncation = Outlier_Dectection(allData, col)
    upper, lower = max(truncation), min(truncation)
    d = upper - lower
    if d == 0:
        print("{} is almost a constant".format(col))
        numerical_var.remove(col)
        continue
    allData[col] = truncation.map(lambda x: (upper - x) / d)
    max_min_standardized[col] = [lower, upper]

allFeatures = list(allData.columns)
allFeatures.remove('result')
x_train = np.matrix(allData[allFeatures])
y_train = np.matrix(allData['result']).T


#进一步将训练集拆分成训练集和验证集。在新训练集上进行参数估计，在验证集上决定最优的参数
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train,test_size=0.4,random_state=9)

#Example: select the best number of units in the 1-layer hidden layer
no_hidden_units_selection = {}
feature_columns = [tf.contrib.layers.real_valued_column("", dimension = x_train.shape[1])]
for no_hidden_units in range(50,101,10):
    print("the current choise of hidden units number is {}".format(no_hidden_units))
    clf0 = tf.contrib.learn.DNNClassifier(feature_columns = feature_columns,
                                          hidden_units=[no_hidden_units, no_hidden_units-10,no_hidden_units-20],
                                          n_classes=2,
                                          dropout = 0.5
                                          )
    clf = SKCompat(clf0)
    clf.fit(x_train, y_train, batch_size=256,steps = 100000)
    #monitor the performance of the model using AUC score
    clf_pred_proba = clf._estimator.predict_proba(x_validation)
    pred_proba = [i[1] for i in clf_pred_proba]
    auc_score = roc_auc_score(y_validation.getA(),pred_proba)
    no_hidden_units_selection[no_hidden_units] = auc_score
best_hidden_units = max(no_hidden_units_selection.items(), key=operator.itemgetter(1))[0]


dropout_selection = {}
feature_columns = [tf.contrib.layers.real_valued_column("", dimension = x_train.shape[1])]
for dropout_prob in np.linspace(0,0.99,20):
    print("the current choise of drop out rate is {}".format(dropout_prob))
    clf0 = tf.contrib.learn.DNNClassifier(feature_columns = feature_columns,
                                          hidden_units = [best_hidden_units, best_hidden_units-10,best_hidden_units-20],
                                          n_classes=2,
                                          dropout = dropout_prob
                                          )
    clf = SKCompat(clf0)
    clf.fit(x_train, y_train, batch_size=256,steps = 100000)
    #monitor the performance of the model using AUC score
    clf_pred_proba = clf._estimator.predict_proba(x_validation)
    pred_proba = [i[1] for i in clf_pred_proba]
    auc_score = roc_auc_score(y_validation.getA(),pred_proba)
    dropout_selection[dropout_prob] = auc_score
best_dropout_prob = max(dropout_selection.items(), key=operator.itemgetter(1))[0]

clf1 = tf.contrib.learn.DNNClassifier(feature_columns = feature_columns,
                                          hidden_units = [best_hidden_units, best_hidden_units-10,best_hidden_units-20],
                                          n_classes=2,
                                          dropout = best_dropout_prob)
clf1.fit(x_train, y_train, batch_size=256,steps = 100000)
clf_pred_proba = clf1.predict_proba(x_train)
pred_proba = [i[1] for i in clf_pred_proba]
auc_score = roc_auc_score(y_train.getA(),pred_proba)
