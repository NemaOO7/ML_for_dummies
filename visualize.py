# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 12:01:51 2022

@author: sabhi
"""

import seaborn as sns
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.linear_model import LinearRegression,LogisticRegression,Ridge,Lasso,RidgeClassifier
from sklearn.svm import SVC,SVR
from sklearn.tree import  DecisionTreeClassifier,DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn import tree
from matplotlib.colors import ListedColormap
import statsmodels.api as sm
import sklearn.metrics
import sklearn
import traceback
import pandas as pd
import logging
import uuid
import pickle
import time
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
class VisualLearningCore:
    def __init__(self,model,X,y=None):
        self.X = X
        self.y = y
        self.model = None
        self.counter =0

        try:
            import matplotlib
            matplotlib.use('qtagg')
            
        except:
            pass
    def pairPlot(self,X,y):
        try:
            df = pd.concat([X, y], axis=1)
            sns.pairPlot(df)
            self.counter =self.counter+1
        except:
            pass
    def contourPlot(self,X,y,model):
        X_set= X.values
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        Xpred = np.array([X1.ravel(), X2.ravel()] + [np.repeat(0.7, X1.ravel().size) for i in range(X_set.shape[1]-2)]).T
    # Xpred now has a grid for x1 and x2 and average value (0) for x3 through x13
        pred = model.predict(Xpred).reshape(X1.shape)   # is a matrix of 0's and 1's !
        plt.contourf(X1, X2, pred,
                 alpha = 0.75, cmap = ListedColormap(('red', 'green')))
        self.counter =self.counter+1
        plt.show()
        plt.savefig("plot1",format='pdf')
    def treeplot(self,X,y,model):
        X_set, y_set = X.values,y.values
        print(X_set)
        y_set=LabelEncoder().fit_transform(y_set)
        print(y_set)
        model=DecisionTreeClassifier()
        model.fit(X,y_set)
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        Xpred = np.array([X1.ravel(), X2.ravel()] + [np.repeat(np.median(X[i+1]), X1.ravel().size) for i in range(X.shape[1]-2)]).T
        # Xpred now has a grid for x1 and x2 and average value (0) for x3 through x13
        pred = model.predict(Xpred).reshape(X1.shape)   # is a matrix of 0's and 1's !
        tree.plot_tree(model)
        try:
            tree.plot_tree(model)
            tree.savefig("plot1",format='pdf')
        except:
            plt.show()
            plt.savefig("plot1",format='pdf')
        
        self.counter =self.counter+1
    def treeplot1(self,X,y,model):
        X_set, y_set = X.values,y.values
        print(X_set)
        y_set=LabelEncoder().fit_transform(y_set)
        print(y_set)
        model=DecisionTreeRegressor()
        model.fit(X,y_set)
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        Xpred = np.array([X1.ravel(), X2.ravel()] + [np.repeat(np.median(X[i+1]), X1.ravel().size) for i in range(X.shape[1]-2)]).T
        # Xpred now has a grid for x1 and x2 and average value (0) for x3 through x13
        pred = model.predict(Xpred).reshape(X1.shape)   # is a matrix of 0's and 1's !
        tree.plot_tree(model)
        try:
            tree.show()
            tree.savefig("plot1",format='pdf')
        except:
            plt.show()
            plt.savefig("plot1",format='pdf')
        self.counter =self.counter+1
    def summuary(self,X,y,model):
        try:
            print(model.summuary())
        except:
            pass
    def dataInfo(self,X,y):
        try:
            df = pd.concat([X, y], axis=1)
            df.description
        except:
            print("1 has occured")
    def all_encoder(self):
        try :
            self.pairPlot(self.X,self.y,self.model)
        except:
            print("2 has occured")
        try :
            self.contourPlot(self.X,self.y,self.model)
        except:
            print("3 has occured")
        
        # try :
        #     self.treeplot1(self.X,self.y,self.model)
        # except:
        #     print("4 has occured")
import seaborn as sns
df = sns.load_dataset('iris')