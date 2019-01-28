import pandas as pd
import datetime
import collections
import numpy as np
import numbers
import random
import sys
import seaborn as sns
import pickle
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
from importlib import reload
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")
from sklearn2pmml import sklearn2pmml,PMMLPipeline
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import SKCompat
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV

def MissingCategorial(df,x):
    missing_vals = df[x].map(lambda x: int(x!=x))
    return sum(missing_vals)*1.0/df.shape[0]

def MissingContinuous(df,x):
    missing_vals = df[x].map(lambda x: int(np.isnan(x)))
    return sum(missing_vals) * 1.0 / df.shape[0]

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
trainData = pd.read_csv('test1.csv')
##只计算统计变量
allFeatures = list(trainData.columns)
allFeatures.remove('Unnamed: 0')
allFeatures.remove('result')

#检查是否有常数型变量，并且检查是类别型还是数值型变量
numerical_var = []
for col in allFeatures:
    if len(set(trainData[col])) == 1:
        print('delete {} from the dataset because it is a constant'.format(col))
        del trainData[col]
        allFeatures.remove(col)
    else:
        uniq_valid_vals = [i for i in trainData[col] if i == i]
        uniq_valid_vals = list(set(uniq_valid_vals))
        if len(uniq_valid_vals) >= 10 and isinstance(uniq_valid_vals[0], numbers.Real):
            numerical_var.append(col)

categorical_var = [i for i in allFeatures if i not in numerical_var]

'''
对类别型变量，如果缺失超过80%, 就删除，否则保留。
'''
missing_pcnt_threshould_1 = 0.8
for col in categorical_var:
    missingRate = MissingCategorial(trainData,col)
    print('{0} has missing rate as {1}'.format(col,missingRate))
    if missingRate > missing_pcnt_threshould_1:
        categorical_var.remove(col)
        del trainData[col]
allData_bk = trainData.copy()

'''
用one-hot对类别型变量进行编码
'''
dummy_map = {}
dummy_columns = []
for raw_col in categorical_var:
    dummies = pd.get_dummies(trainData.loc[:, raw_col], prefix=raw_col)
    col_onehot = pd.concat([trainData[raw_col], dummies], axis=1)
    col_onehot = col_onehot.drop_duplicates()
    trainData = pd.concat([trainData, dummies], axis=1)
    del trainData[raw_col]
    dummy_map[raw_col] = col_onehot
    dummy_columns = dummy_columns + list(dummies)


'''
检查数值型变量
'''
missing_pcnt_threshould_2 = 0.8
deleted_var = []
for col in numerical_var:
    missingRate = MissingContinuous(trainData, col)
    print('{0} has missing rate as {1}'.format(col, missingRate))
    if missingRate > missing_pcnt_threshould_2:
        deleted_var.append(col)
        print('we delete variable {} because of its high missing rate'.format(col))
    else:
        if missingRate > 0:
            not_missing = trainData.loc[trainData[col] == trainData[col]][col]
            missing_position = trainData.loc[trainData[col] != trainData[col]][col].index
            not_missing_sample = random.sample(list(not_missing), len(missing_position))
            trainData.loc[missing_position,col] = not_missing_sample
            #del allData[col]
            #allData[col] = makeuped
            missingRate2 = MissingContinuous(trainData, col)
            print('missing rate after making up is:{}'.format(str(missingRate2)))

if deleted_var != []:
    for col in deleted_var:
        numerical_var.remove(col)
        del trainData[col]


'''
对极端值变量做处理。
'''
max_min_standardized = {}
for col in numerical_var:
    truncation = Outlier_Dectection(trainData, col)
    upper, lower = max(truncation), min(truncation)
    d = upper - lower
    if d == 0:
        print("{} is almost a constant".format(col))
        numerical_var.remove(col)
        continue
    trainData[col] = truncation.map(lambda x: (upper - x)/d)
    max_min_standardized[col] = [lower, upper]


all_features = list(trainData.columns)
all_features.remove('result')
X_train, y_train = trainData[all_features], trainData['result']

param_test1 = {'max_depth':range(3,10,2), 'min_child_weight':range(1,6,2)}
gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=100, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                  objective= 'binary:logistic',nthread=4,scale_pos_weight=1,seed=27),
                        param_grid = param_test1,scoring='roc_auc',n_jobs=4,iid=False,cv=5)
gsearch1.fit(X_train,y_train)
best_max_depth, best_min_child_weight = gsearch1.best_params_['max_depth'],gsearch1.best_params_['min_child_weight']

param_test2 = {'gamma':[i/10.0 for i in range(0,5)]}
gsearch2 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=100, subsample=0.8, colsample_bytree=0.8, max_depth= best_max_depth,
                                                  min_child_weight=best_min_child_weight, objective= 'binary:logistic',nthread=4,scale_pos_weight=1,seed=27),
                        param_grid = param_test2,scoring='roc_auc',n_jobs=4,iid=False,cv=5)
gsearch2.fit(X_train,y_train)
best_gamma = gsearch2.best_params_['gamma']

param_test3 = {'subsample':[i/10.0 for i in range(6,10)],'colsample_bytree':[i/10.0 for i in range(6,10)]}
gsearch3 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=100, max_depth= best_max_depth, gamma=best_gamma,
                                                  min_child_weight=best_min_child_weight, objective= 'binary:logistic',nthread=4,scale_pos_weight=1,seed=27),
                        param_grid = param_test3,scoring='roc_auc',n_jobs=4,iid=False,cv=5)
gsearch3.fit(X_train,y_train)
best_colsample_bytree, best_subsample = gsearch3.best_params_['colsample_bytree'], gsearch3.best_params_['subsample']

param_test4 = {'reg_alpha':[0.01,0.1,1,10,50,100,200,500]}
gsearch4 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=100, max_depth= best_max_depth, gamma=best_gamma,
                                                  colsample_bytree = best_colsample_bytree, subsample = best_subsample,
                                                  min_child_weight=best_min_child_weight, objective= 'binary:logistic',nthread=4,scale_pos_weight=1,seed=27),
                        param_grid = param_test4,scoring='roc_auc',n_jobs=4,iid=False,cv=5)
gsearch4.fit(X_train,y_train)
best_reg_alpha = gsearch4.best_params_['reg_alpha']


param_test5 = {'n_estimators':range(100,401,10)}
gsearch5 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1,  max_depth= best_max_depth, gamma=best_gamma,
                                                  colsample_bytree = best_colsample_bytree, subsample = best_subsample,reg_alpha=best_reg_alpha,
                                                  min_child_weight=best_min_child_weight, objective= 'binary:logistic',nthread=4,scale_pos_weight=1,seed=27),
                        param_grid = param_test5,scoring='roc_auc',n_jobs=4,iid=False,cv=5)
gsearch5.fit(X_train,y_train)
best_n_estimators = gsearch5.best_params_

#用获取得到的最优参数再次训练模型
best_xgb = XGBClassifier(learning_rate =0.1, n_estimators=100, max_depth= best_max_depth, gamma=best_gamma,
                         colsample_bytree = best_colsample_bytree, subsample = best_subsample, reg_alpha=best_reg_alpha,
                         min_child_weight=best_min_child_weight, objective= 'binary:logistic',nthread=4,scale_pos_weight=1,seed=27)
best_xgb.fit(X_train,y_train)
y_pred = best_xgb.predict_proba(X_train)[:,1]
roc_auc_score(y_train, y_pred)
feature_importance = best_xgb.feature_importances_

#利用特征重要性筛去一部分无用的变量
X_train_temp = X_train.copy()
features_in_model = all_features
while(min(feature_importance)<0.00001):
    features_in_model = [features_in_model[i] for i in range(len(feature_importance)) if feature_importance[i] > 0.00001]
    X_train_temp= X_train_temp[features_in_model]
    best_xgb.fit(X_train_temp, y_train)
    feature_importance = best_xgb.feature_importances_

y_pred = best_xgb.predict_proba(X_train_temp)[:,1]
print(roc_auc_score(y_train, y_pred) ) 
print('There are {} features in the raw data'.format(X_train.shape[1]))
print('There are {} features in the reduced data'.format(X_train_temp.shape[1]))