# -*- coding= utf-8 -*-
"""
Created on Tue Mar 15 21=39=15 2022

@author= nemam
"""


from sklearn.linear_model import LinearRegression,LogisticRegression,Ridge,Lasso,RidgeClassifier
from sklearn.svm import SVC,SVR
from sklearn.tree import  DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import mean_squared_error,accuracy_score,r2_score,make_scorer
import sklearn.metrics
import sklearn
import traceback
import pandas as pd
def for_everyone():
    '''
    create new env
    after that only use conda install to install package
    note the name of package so every one can install that
    my-package are-
    conda install -c anaconda statsmodels
    conda install -c conda-forge tensorflow
    conda install -c anaconda scikit-learn
    conda install -c anaconda seaborn

    Returns
    -------
    None.

    '''
def suggestion():
    '''
    1-add extra parameter of input that is two blank block ____ : ____ first name of first custom parameter.seconds value of custom parameter
    this will increace flexability of project by a lot do it for every parameter
    2- tosave fucntion use a prompt to choosice file path and file name same for toload
    Returns
    -------
    None.

    '''
def SVLearningCore(LinearRegression,LogisticRegression,Ridge,Lasso,RidgeClassifie, SVC,SVR,RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier,RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,KNeighborsClassifier,KNeighborsRegressor):
    '''  
    SVLearningCore: model_para_list : TYPE Dictionary
            DESCRIPTION.[Must include mertics]Main Dictionary contain model name. SubDictionary of Hyparamenter key=name of hyper parameter & value=values of hypermeter 
            Example-{"SVR":{"kernel":[ "linear", "poly", "rbf", "sigmoid"],"C":[0.3,0.9,1.3]}}
    Parameters
    ----------
    LinearRegression : fit_intercept =True,normalize =False,copy_X = True
        DESCRIPTION. Regression without regularization
    LogisticRegression : penalty= [ 'l1', 'l2', 'elasticnet',none],C = [0.01,0.06,0.1,0.6,1.2,1.6]
        DESCRIPTION. LogisticRegression
    Ridge : TYPE alpha= [0.01,0.1,1,5,10,0.5],solver['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],tol=[0.01,0.06,0.1,0.6,1.2,1.6]
        DESCRIPTION.
    Lasso : TYPE alpha= [0.01,0.1,1,5,10,0.5],tol=[0.01,0.06,0.1,0.6,1.2,1.6],precompute[True , False]
        DESCRIPTION.
    RidgeClassifie : TYPE alpha= [0.01,0.1,1,5,10,0.5],solver='auto',tol=[0.01,0.06,0.1,0.6,1.2,1.6]
        DESCRIPTION.
    SVC :TYPE   C=  [0.01,0.06,0.1,0.6,1.2,1.6],kernel ['linear', 'poly', 'rbf', 'sigmoid'],
        DESCRIPTION.
    SVR : TYPE  C=  [0.01,0.06,0.1,0.6,1.2,1.6],kernel ['linear', 'poly', 'rbf', 'sigmoid'],epsilon = [0.01,0.06,0.1,0.6,1.2,1.6]
        DESCRIPTION.
    RandomForestClassifier : TYPE criterion["gini", "entropy"],'bootstrap'= [True, False],'max_depth'= [10, 30, 40, 90, 100, None],'max_features'= ['auto', 'sqrt'],'min_samples_leaf'= [1, 2, 4],'min_samples_split'= [2, 5, 10],'n_estimators'= [200,  600, 800,  1600, 2000]
        DESCRIPTION. 
    GradientBoostingClassifier : TYPE  learning_rate= [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],min_samples_split= np.linspace(0.1, 0.5, 12),"min_samples_leaf"= np.linspace(0.1, 0.5, 12),"max_depth"=[3,8,15],"max_features"=["log2","sqrt"],"criterion"= ["friedman_mse",  "mae"],subsample=[0.5, 0.618, 0.8, 0.9, 0.95, 1.0],n_estimators=[10,20,60,100]{import numpy}
        DESCRIPTION.
    AdaBoostClassifier : TYPE learning_rate= [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],n_estimators=[10,20,60,100]
        DESCRIPTION.
    RandomForestRegressor :TYPE criterion["gini", "entropy"],bootstrap= [True, False],max_depth= [10, 30, 40, 90, 100, None],max_features= ['auto', 'sqrt'],min_samples_leaf= [1, 2, 4],min_samples_split= [2, 5, 10],n_estimators= [200,  600, 800,  1600, 2000]
        DESCRIPTION.
    GradientBoostingRegressor : TYPE learning_rate= [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],min_samples_split"= np.linspace(0.1, 0.5, 12),"min_samples_leaf"= np.linspace(0.1, 0.5, 12),"max_depth"=[3,8,15],"max_features"=["log2","sqrt"],"criterion"= ["friedman_mse",  "mae"],"subsample"=[0.5, 0.618, 0.8, 0.9, 0.95, 1.0],"n_estimators"=[10,20,60,100]{import numpy}
        DESCRIPTION.
    AdaBoostRegressor : TYPE learning_rate= [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],n_estimators=[10,20,60,100]
        DESCRIPTION.
    KNeighborsClassifier : TYPE 'n_neighbors'= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,13, 14, 15, 16, 17, 18, 19, 20, 21, 22,23, 24, 25, 26, 27, 28, 29, 30],p=[1,2,5]
        DESCRIPTION.
    KNeighborsRegressor : TYPE 'n_neighbors'= [2,3,4,5,6,7,8,9,10,11,12,13,14,15], 'weights'= ['uniform','distance'],'p'=[1,2,5]
        DESCRIPTION.
    
    Returns
    -------
    None.

    '''    
    pass
def all_metrics():
    '''
    __help_score() return all metrics for scoreing
    '''
def Inferential():
    '''
    Note all information is in the class itself

    Returns
    -------
    None.

    '''
    pass
def USLearningCore():
     '''
    Note all information is in the class itself
    ISE MODULE ME INPUT LIYA HE EK JAGAHE tere ko uske jaga he phele ek graph dekha na he and then ek input lena he ye do methode me karna he dono __ se suru hoti he 

    Returns
    -------
    None.

    '''
    
def DReductionCore():
    '''
    Note all information is in the class itself

    Returns
    -------
    None.

    '''
    pass
    