import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation, metrics
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model.logistic import LogisticRegression
import warnings
warnings.filterwarnings("ignore")
'''
之前所需要进行的分箱或者预处理操作暂略
'''

def KS(df, score, target):
    '''
    :param df: 包含目标变量与预测值的数据集
    :param score: 得分或者概率
    :param target: 目标变量
    :return: KS值
    '''
    total = df.groupby([score])[target].count()
    bad = df.groupby([score])[target].sum()
    all = pd.DataFrame({'total':total, 'bad':bad})
    all['good'] = all['total'] - all['bad']
    all[score] = all.index
    all = all.sort_values(by=score,ascending=False)
    all.index = range(len(all))
    all['badCumRate'] = all['bad'].cumsum() / all['bad'].sum()
    all['goodCumRate'] = all['good'].cumsum() / all['good'].sum()
    KS = all.apply(lambda x: x.badCumRate - x.goodCumRate, axis=1)
    return max(KS)

#读取数据
trainData = pd.read_csv('test1.csv')
##只计算统计变量
allFeatures = list(trainData.columns)
allFeatures.remove('Unnamed: 0')
allFeatures.remove('result')


trainData, testData = train_test_split(trainData,test_size=0.4,random_state=0)

X = np.matrix(trainData[allFeatures])
y = trainData['result']


# 未经调参进行GBDT模型训练
gbm0 = GradientBoostingClassifier(random_state=10)
gbm0.fit(X,y)

y_pred = gbm0.predict(X)
y_predprob = gbm0.predict_proba(X)[:,1].T
trainData['probb'] = y_predprob
print ("Accuracy : %.4g" % metrics.accuracy_score(y, y_pred)) # 0.8919
print ("AUC Score (Train): %f" % metrics.roc_auc_score(np.array(y.T), y_predprob)) # 0.
print ("KS is :{}".format(KS(trainData, 'probb', 'gbflag')))

X_test = np.matrix(testData[allFeatures])
y_test = np.matrix(testData['result']).T
#在测试集上测试GBDT性能
y_predtest = gbm0.predict(X_test)
y_predprobtest = gbm0.predict_proba(X_test)[:,1].T
testData['predprob'] = list(y_predprobtest)
print ("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_predtest))
print ("AUC Score (Test): %f" % metrics.roc_auc_score(np.array(y_test)[:,0], y_predprobtest))
print ("KS is :{}".format(KS(testData, 'predprob', 'gbflag')))



'''
GBDT调参
'''
# 1, 选择较小的步长(learning rate)后，对迭代次数(n_estimators)进行调参

X = pd.DataFrame(X)

param_test1 = {'n_estimators':range(20,81,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=30,
                                  min_samples_leaf=5,max_depth=8,max_features='sqrt', subsample=0.8,random_state=10),
                       param_grid = param_test1, scoring='roc_auc',iid=False,cv=5)
gsearch1.fit(X,y)
best_n_estimator = gsearch1.best_params_['n_estimators']


# 2, 对决策树最大深度max_depth和内部节点再划分所需最小样本数min_samples_split进行网格搜索
param_test2 = {'max_depth':range(3,16), 'min_samples_split':range(2,10)}
gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=best_n_estimator, min_samples_leaf=20, max_features='sqrt', subsample=0.8, random_state=10),
                        param_grid = param_test2, scoring='roc_auc',iid=False, cv=5)
gsearch2.fit(X,y)
best_max_depth = gsearch2.best_params_['max_depth']



#3, 再对内部节点再划分所需最小样本数min_samples_split和叶子节点最少样本数min_samples_leaf一起调参
param_test3 = {'min_samples_split':range(10,101,10), 'min_samples_leaf':range(5,51,5)}
gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=best_n_estimator,max_depth=best_max_depth,
                                     max_features='sqrt', subsample=0.8, random_state=10),
                       param_grid = param_test3, scoring='roc_auc',iid=False, cv=5)
gsearch3.fit(X,y)
best_min_samples_split, best_min_samples_leaf = gsearch3.best_params_['min_samples_split'],gsearch3.best_params_['min_samples_leaf']



#4, 对最大特征数max_features进行网格搜索,注意max_features 必须小于最大的树值
param_test4 = {'max_features':range(5,best_min_samples_leaf+1,5)}
gsearch4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=best_n_estimator,max_depth=best_max_depth, min_samples_leaf =best_min_samples_leaf,
               min_samples_split =best_min_samples_split, subsample=0.8, random_state=10),
                       param_grid = param_test4, scoring='roc_auc',iid=False, cv=5)
gsearch4.fit(X,y)
best_max_features = gsearch4.best_params_['max_features']

#5, 对采样比例进行网格搜索
param_test5 = {'subsample':[0.6+i*0.05 for i in range(8)]}
gsearch5 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=best_n_estimator,max_depth=best_max_depth,
                                                               min_samples_leaf =best_min_samples_leaf, max_features=best_max_features,random_state=10),
                       param_grid = param_test5, scoring='roc_auc',iid=False, cv=5)
gsearch5.fit(X,y)
best_subsample = gsearch5.best_params_['subsample']

gbm_best = GradientBoostingClassifier(learning_rate=0.1, n_estimators=best_n_estimator,max_depth=best_max_depth,
                                      min_samples_leaf =best_min_samples_leaf, max_features=best_max_features,subsample =best_subsample, random_state=10)
gbm_best.fit(X,y)

#在测试集上测试并计算性能
y_pred = gbm_best.predict(X_test)
y_predprob = gbm_best.predict_proba(X_test)[:,1].T
testData['predprob'] = list(y_predprob)
print ("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))
print ("AUC Score (Test): %f" % metrics.roc_auc_score(np.array(y_test)[:,0], y_predprob))
print ("KS is :{}".format(KS(testData, 'predprob', 'gbflag')))


#GBDT模型生成特征

# 利用GBDT的结果衍生新的特征
new_feature= gbm_best.apply(X)[:,:,0]
grd_enc = OneHotEncoder()
grd_enc.fit(new_feature)
x = grd_enc.transform(new_feature)
classifier=LogisticRegression()
classifier.fit(x,y)

new_feature_test= gbm_best.apply(X_test)[:,:,0]
x_test = grd_enc.transform(new_feature_test)
y_pred_lr = classifier.predict_proba(x_test)[:,1]
lr_pred = pd.DataFrame({'predprob':y_pred_lr, 'gbflag': np.array(y_test)[:,0]})

print ("AUC Score (Test): %f" % metrics.roc_auc_score(np.array(y_test)[:,0], y_pred_lr))
print ("KS is :%f"%KS(lr_pred, 'predprob', 'gbflag'))

# 将模型保存
saveModel = open('./model/GBDT_statsmodels_GBDT.pkl', 'wb+')
pickle.dump(gbm_best, saveModel)
saveModel.close()

