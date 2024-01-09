# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 19:20:03 2022

@author: nemam
"""
import pandas as pd
from sklearn.decomposition import  KernelPCA
import pickle
from xgboost import XGBClassifier

def model_predictor(X,y):
    temp = None
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    try:
        desz_pad = [0]
        desz1 =[0]*(8*25)  
        
        
        df = pd.concat([X,y],axis=1)
        if len(df.columns)>=25:
              pca = KernelPCA(n_components=25)
              df = pd.DataFrame(pca.fit_transform(df))
        desz1 =[0]*(8*25)  
        count=0
        for (columnName, columnData) in df.describe().iteritems():
           for j in  (columnData.values):
               desz1[count] = j 
               count+=1
        
        count=0
        desz2 = [0]*(8*2)  #pca component summary
        pca = KernelPCA(n_components=2)
        tempkpca = pd.DataFrame(pca.fit_transform(df))
        for (columnName, columnData) in tempkpca.describe().iteritems():
                for j in  (columnData.values):
                   desz2[count] = j 
                   count+=1
        
        
        count=0
        corr = df.corr().to_numpy()
        desz3 = [0]*((25*25)//2)
        for i in range(corr.shape[0]):
            for j in  range(corr.shape[1]):
                   if i < j:
                       desz3[count] = corr[i][j]
                       count+=1
        desz4 = [0]
        reverese_dic = {16: 'KNeighborsRegressor',
         15: 'AdaBoostRegressor',
         14: 'GradientBoostingRegressor',
         13: 'RandomForestRegressor',
         12: 'DecisionTreeRegressor',
         11: 'SVR',
         10: 'Lasso',
         9: 'Ridge',
         8: 'LinearRegression',
         7: 'KNeighborsClassifier',
         6: 'AdaBoostClassifier',
         5: 'GradientBoostingClassifier',
         4: 'RandomForestClassifier',
         3: 'DecisionTreeClassifier',
         2: 'SVC',
         1: 'LogisticRegression',
         0: 'RidgeClassifier'
         }
        temp = reverese_dic[int(pickle.load(open(r"XG7839.sav", 'rb')).predict(pd.DataFrame([desz1+desz2+desz3+desz4+desz_pad])))]
        if temp is None:
            return 'SVC'
        else:
            return temp
       
    except Exception as e:
        print(e)
    
       
    


# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# import seaborn as sns
# from xgboost import __version__
# ##vers = LooseVersion(__version__)
# print(__version__)
# df = sns.load_dataset('iris')

# X = sns.load_dataset('iris')[["sepal_length" ,"sepal_width","petal_length"]]
# y = sns.load_dataset('iris')['species']
# y = LabelEncoder().fit_transform(y)
# model_predictor(X,y)
# print(model_predictor(X,y))
