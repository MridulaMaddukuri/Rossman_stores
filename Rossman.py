
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import ctypes
import copy
import sys
from sklearn.metrics import zero_one_loss
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import coo_matrix, hstack, csr_matrix, vstack
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import ExtraTreesClassifier
from pyfm import pylibfm
# from adaboost_multiple import AdaBoost
from itertools import combinations
from sklearn import metrics, cross_validation, linear_model
# from logistic_regression_updated import group_data,OneHotEncoder2
from sklearn.cross_validation import StratifiedKFold
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import itertools
from datetime import datetime


from sklearn.base import TransformerMixin
class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.fill)


# In[2]:

def load_data(filetrain,filetest,filestore):
    '''Function to load train and test data into pandas data frame. 
    Argument1 : training dataset filename
    Argument2 : test dataset filename
    '''
    train_df = pd.read_csv(filetrain, header=0)
    test_df = pd.read_csv(filetest, header=0)
    store_df = pd.read_csv(filestore, header=0)

    return train_df, test_df, store_df

train_df, test_df, store_df = load_data('train.csv','test.csv','store.csv')
train_df = train_df.merge(right = store_df, how = 'inner', on = 'Store')
test_df = test_df.merge(right = store_df, how = 'inner', on = 'Store')


# In[3]:

train_df['Date'] = train_df['Date'].map(lambda x : datetime.strptime(x, '%Y-%m-%d'))
test_df['Date'] = test_df['Date'].map(lambda x : datetime.strptime(x, '%Y-%m-%d'))
train_df['Year'] = train_df['Date'].map(lambda x : int(x.year))
train_df['Month'] = train_df['Date'].map(lambda x : int(x.month))
test_df['Year'] = test_df['Date'].map(lambda x : int(x.year))
test_df['Month'] = test_df['Date'].map(lambda x : int(x.month))
test_df = test_df[[col for col in test_df.columns if col not in ['Date']]]
train_df = train_df[[col for col in train_df.columns if col not in ['Date']]]


# In[ ]:

train_X = train_df[[col for col in train_df.columns if col not in ['Customers','Sales']]]
test_X = test_df
train_Y = train_df['Sales']
Big_X = train_X.append(test_X)
Big_Imputed = DataFrameImputer().fit_transform(Big_X)
train_X = Big_Imputed.iloc[0:len(train_df)]
test_X = Big_Imputed.iloc[len(train_df):len(Big_Imputed)]
train_X = train_X[[col for col in train_X.columns if col not in ['Id']]]
test_X = test_X[[col for col in test_X.columns if col not in ['Id']]]


# In[ ]:

def convert_to_num(df, column):
    x = np.unique(df[column])
    dic = {}
    count = 0
    for i in range(len(x)):
        dic[x[i]] = i
    df[column] = df[column].map(lambda x : dic[x])
    return df

Big_X = train_X.append(test_X)

Big_X = convert_to_num(Big_X, 'Assortment')
Big_X = convert_to_num(Big_X, 'PromoInterval')
Big_X = convert_to_num(Big_X, 'StoreType')
Big_X = convert_to_num(Big_X, 'StateHoliday')
train_X = Big_X.iloc[0:len(train_df)]
test_X = Big_X.iloc[len(train_df):len(Big_X)]


# In[ ]:

# gbm = xgb.XGBRegressor().fit(train_X.as_matrix(), train_Y.as_matrix())
param_test1 = {'max_depth':range(6,13,3),'n_estimators' : [100,200], 'colsample_bytree' : [0.4,0.8]}
gsearch1 = GridSearchCV(estimator = xgb.XGBRegressor(learning_rate =0.05, n_estimators=300, max_depth=5,min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,scale_pos_weight=1, seed=11), param_grid = param_test1,n_jobs=20,iid=False, cv=5)
gsearch1.fit(train_X,train_Y)
bp = gsearch1.best_params_
print bp


# In[ ]:

prediction = gbm.predict(test_X)
submission = pd.DataFrame(np.vstack([test_df['Id'].values,prediction]).T, columns = ['Id','Sales'])


# In[ ]:

submission['Sales'][test_df['Open'] == 0] = 0
submission.to_csv('submission1.csv')


# In[ ]:




# In[ ]:



