# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 13:30:54 2022

@author: nemam
import sys
"""


from sklearn.linear_model import LinearRegression,LogisticRegression,Ridge,Lasso,RidgeClassifier
from sklearn.svm import SVC,SVR
from sklearn.tree import  DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import mean_squared_error,accuracy_score,r2_score,make_scorer
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
from sklearn.model_selection import train_test_split
class SVLearningCore:
    '''
        initialinitialization of parameters

        Parameters
        ----------
         X : TYPE DataFrame
            DESCRIPTION.Contain all feature of dataset 
        y : TYPE DataFrame Vector
            DESCRIPTION. Traget variable
        model_para_list : TYPE Dictionary
            DESCRIPTION.[Must include mertics]Main Dictionary contain model name. SubDictionary of Hyparamenter key=name of hyper parameter & value=values of hypermeter 
            Example-{"SVR":{"kernel":[ "linear", "poly", "rbf", "sigmoid"],"C":[0.3,0.9,1.3]}}
        score : TYPE Variable
            DESCRIPTION.How to measure and score model.----MUST----
        rcflag : TYPE Boolean, optional
            DESCRIPTION. The default is False.false for regression true for classifcation       
        utime_mode: TYPE, optional
            DESCRIPTION.The default is 'low'.Time mode how much time will it take weather to try all combination or to apply random combination of hyperparameter along with iter
        cv : TYPE Int, optional
            DESCRIPTION.The default is 3.number of cross valdition.
        seed : TYPE Int
            DESCRIPTION. The default is 100.Number of iterationfor random hyperameter tuning.
        iterr : TYPE Int, optional
            DESCRIPTION.The default is 10.Number of iteration for low time config.
        n_jobs : TYPE Int, optional
            DESCRIPTION.1 The default is 1.represent linear -1 represent prallelism
        __mylogconfig() : TYPE void
            DESCRIPTION. initilize logging system
        __rchecker() : TYPE void
            DESCRIPTION. regresser classifer check
        __defaultscore(): TYPE void
            DESCRIPTION. default value initilizer

        Returns
        -------
        None.

        '''

    def __init__(self,X,y,model_para_list,score,rcflag=False,utime_mode='low',cv=3,seed=100,iterr=10,n_jobs=-1):
        
        
        
        self.X = X
        self.y = y
        self.umodel_para_list =model_para_list 
        self.utime_mode = utime_mode
        self.cv = cv
        self.modelframe = None
        self.score= score
        self.iter = iterr  
        self.seed = seed
        self.n_jobs = n_jobs
        self.rcflag = rcflag
        self.bindedmodel = {}#IGNODRE
        self.__mylogconfig()
        self.__rchecker()
        self.__defaultscore()
        self.id = uuid.uuid1()
        logging.info('construtor completed successfully')
    def __mylogconfig(self):
        '''
        configure logging module

        Returns
        -------
        None.

        '''
        logging.basicConfig(filename="SVLearningcore.log",level=logging.INFO,format="[ %(levelname)s :  %(asctime)s :"+__name__+" : %(funcName)17s() -%(lineno)d ] %(message)s")
        logging.info("Logging new SESSION")
        
    def predict(self,model):
        '''
    
        if model is created and only predition are moade

        Parameters
        ----------
        model : TYPE sklearn model
            DESCRIPTION. sklean supervisemodel


        Returns
        -------
        temp : TYPE tuple(X featuer,y predication)
            DESCRIPTION. feature and predication

        '''
        try:
            temp =  (self.X,model.predict(model))
        except Exception as e:
            logging.critical("either model or data are incorrent "+e)
        logging.info('def predict completed successfully')
        return temp
    
    def __datasetcreater(self):
        '''
        This funcition is use to create dataset so that AI can learn

        Returns
        -------
        None.

        '''
        new_model = self.modelframe.copy()
        try:
            desz_pad = [0]
            desz1 =[0]*(8*25)  
            
            if self.model is not None:
                desz0 = [str(self.id)]
                df = pd.concat([pd.DataFrame(self.X),pd.DataFrame(self.y)],axis=1)
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
                desz4 = [1 if self.rcflag else 0]
                model_dict = {
                 'RidgeClassifier': 0,
                 'LogisticRegression': 1,
                 'SVC': 2,
                 'DecisionTreeClassifier': 3,
                 'RandomForestClassifier': 4,
                 'GradientBoostingClassifier': 5,
                 'AdaBoostClassifier': 6,
                 'KNeighborsClassifier': 7,
                 'LinearRegression': 8,
                 'Ridge': 9,
                 'Lasso': 10,
                 'SVR': 11,
                 'DecisionTreeRegressor': 12,
                 'RandomForestRegressor': 13,
                 'GradientBoostingRegressor': 14,
                 'AdaBoostRegressor': 15,
                 'KNeighborsRegressor': 16
                 }
                
                desz5 = [model_dict[str(new_model.iloc[0][0])]]

        except Exception as e:
                logging.error("dataset went wrong "+str(e)+traceback.format_exc())
        else:
            try:
                
                
                for i,j in new_model['best_params'][0].items():
                    new_model.insert(2,i,j)
                del new_model['best_params']
                
                pd.DataFrame([new_model.iloc[0,:]]).to_csv(r"C:\Users\nemam\Untitled Folder\Major\_specificdataset.csv",sep =',',index = False,mode='a+',header=False)
                new_model.to_csv(r"C:\Users\nemam\Untitled Folder\Major\_logdataset.csv",sep =',',index = False,mode='a+',header=False)
                pd.DataFrame([desz0+desz1+desz2+desz3+desz4+desz_pad+desz5]).to_csv(r"C:\Users\nemam\Untitled Folder\Major\_dataset.csv",index = False,mode='a+',header=False)
                logging.info("DATA WRTITTEN SUCCEsS FULLIY")
            except Exception as e:
                    logging.error("read write error "+str(e)+traceback.format_exc())
        logging.info('__datasetcreater completed successfully')
        new_model = None
    def tosave(self):
        '''
        Save model on to drive

        Returns
        -------
        None.

        '''
        try:
            filename = 'finalized_model'+ self.id +'.sav'
            pickle.dump(self.model, open(filename, 'wb'))
        except Exception as e:
            print('file type error'+e)
        logging.info('tosave completed successfully')
        
    def toload(self,filename):
        '''
        Load model from disk

        Parameters
        ----------
        filename : TYPE String
            DESCRIPTION. name of file conatining model

        Returns
        -------
        None.

        '''
        try:
            loaded_model = pickle.load(open(filename, 'rb'))
            self.model = loaded_model
        except Exception as e:
            print('file type error'+e)
        logging.info('toload completed successfully')
            
    def __rchecker(self):
        '''
        Check weather the given algorithm by user and and what the user ask match i.e Regression algo give by user are same as flag set by user i.e rsflag

        Raises
        ------
        Exception
            DESCRIPTION. rsflag are different from given algo

        Returns
        -------
        None.

        '''
        try:
            for i,j in self.umodel_para_list.items():
               if not self.rcflag:
                   if i not in['LinearRegression','LogisticRegression','Ridge','Lasso','DecisionTreeRegressor','SVR','KNeighborsRegressor','RandomForestRegressor','GradientBoostingRegressor','AdaBoostRegressor']:
                       raise Exception("Regression was selected but classification model was given"+i)
               else:
                   if i not in['KNeighborsClassifier','RandomForestClassifier','GradientBoostingClassifier','AdaBoostClassifier','DecisionTreeClassifier','SVC','LogisticRegression','RidgeClassifier']:
                       raise Exception("classification was selected but Regression model was given"+i)
        except Exception as e:
            logging.error(str(e)+traceback.format_exc())
        logging.info('rchecker completed successfully')
            
            
    def __defaultscore(self):
        '''
        set default value of score.

        Returns
        -------
        None.

        '''
        if self.score is None:
            if not self.rcflag:
                self.score = 'r2'
            else:
                self.score = make_scorer(accuracy_score)
        logging.info('defaultscore completed successfully')
        
    @staticmethod
    def __help_score():
        '''
        return all possible scorering technique

        Returns
        -------
        None.

        '''
        return (sklearn.metrics.SCORERS.keys())
    
    def __modelCreater(self,model_name,param = None):
        '''
        Machine Learning Object Creation

        Parameters
        ----------
        model_name : TYPE String
            DESCRIPTION.Name of Machine Leaerning.Must match the Classname.
        param : TYPE String
            DESCRIPTION.Name internal param.Must match the Classname.

        Returns
        -------
        TYPE
            DESCRIPTION.Return corrosponding Machine learning Object.

        '''
        
        try:
            if param is None:
                logging.info(f"{model_name},{param}")
                model = eval(f"{model_name}()")
            else:
                logging.info(f"{model_name},{param}")
                model = eval(f"{model_name}({param})")
        except NameError as e:
            logging.error(str(e)+traceback.format_exc())
        except  Exception as e:
            logging.critical(str(e)+traceback.format_exc())
        else:
            logging.info("Successfull model created")
            return model 
            
    def __hyperbind(self):
        '''
        Bind the machine learning Object to its hyperParameter.

        Returns
        -------
        None.

        '''
        try:
            for i in self.umodel_para_list.keys():
                self.bindedmodel[i] = {"model" : self.__modelCreater(i),"param":self.umodel_para_list[i]}
        except Exception as e :
           logging.critical(str(e)+traceback.format_exc())
        else:
            logging.info("hyperbind done Successfully")
            
    @staticmethod
    def isfloat(num):
        '''
        utility function

        Parameters
        ----------
        num : TYPE String
            DESCRIPTION.String to be checked

        Returns
        -------
        bool
            DESCRIPTION. true if float else false

        '''
        try:
            float(num)
            return True
        except ValueError:
            return False  
    
    def __modeltuner(self):
        '''
        Actucal Machine Learning Take Place Along with HyperParameter Tuning based on time mode.

        Returns
        -------
        TYPE Dataframe 
            DESCRIPTION.Return a DataFrame containing 'model','best_score','best_params','score'.

        '''
        try:
            self.metadata = []
            if self.utime_mode == "high":
                for model_name, mp in self.bindedmodel.items():
                    clf = GridSearchCV(mp["model"],mp["param"],cv =self.cv,scoring=self.score,return_train_score=False,verbose=20,n_jobs=-1)
                    clf.fit(self.X,self.y)
                    self.metadata.append({
                        'id' : self.id,
                        'model':model_name,
                        'best_score': clf.best_score_,
                        'best_params': clf.best_params_,
                        'mode':self.utime_mode,
                        'score':self.score
                        })
                
            else:
                for model_name, mp in self.bindedmodel.items():
                    clf = RandomizedSearchCV(mp["model"],mp["param"],cv =self.cv,scoring=self.score,return_train_score=False,n_iter=self.iter,verbose=20)
                    clf.fit(self.X,self.y)
                    self.metadata.append({
                        'id' : self.id,
                        'model':model_name,
                        'best_score': clf.best_score_,
                        'best_params': clf.best_params_,
                        'mode':self.utime_mode,
                        'score':self.score
                        })
        except TypeError as e:
            logging.error("invalid Paramter\t"+e)
        except ValueError as e:
            logging.error("invalid metric or invalaid value of parameter\t"+str(e)+traceback.format_exc())
        except Exception as e :
           logging.critical(e)
        self.modelframe =  pd.DataFrame(self.metadata,columns=['model','best_score','best_params','score'])
        logging.info("Traning & tuning done Successfully--")
        return self.modelframe
    
    def startlearning(self):
        '''
        User interative function execute creation of machine learning object, binding of object to its parameter, Tuning the HyperParameter. 

        Returns
        -------
        TYPE Dataframe 
            DESCRIPTION.Return a TUPLE(DataFrame containing 'model','best_score','best_params','score',final model)


        '''
        try:
            self.__hyperbind()
            modelframe = self.__modeltuner() 
            self.model = self.train_get(self.X,self.y)
            self.__datasetcreater()
            logging.info('all process almost completed successfully')
            return (modelframe,self.model)
        except Exception as e :
           logging.critical(str(e)+str(traceback.format_exc()))

    def train_get(self,X,y,model=None,param = None):
        '''
        Note- if you want to create model without parameter pass empty string to param

        Parameters
        ----------
        X : TYPE DataFrame
            DESCRIPTION. feature metrix
        y : TYPE DataFrame Vector
            DESCRIPTION.
        model : TYPE STRING, optional
            DESCRIPTION. The default is None.model on which machine learning is to be performed.if None & startLearning is executed then best model is selected
        param : TYPE DICTIONARY, optional
            DESCRIPTION. The default is None.paramter on thich model is to be performed.if None & startLearning is executed then best model tuned hyperparamter is selected

        Raises
        ------
        Exception
            DESCRIPTION. extra or insufficent parameter

        Returns
        -------
        None.

        '''
        
        try:
            if model is None and param is None and len(self.bindedmodel)==0:
                raise Exception("Not sufficent parameter")
            if(model is None and param is None):
                self.modelframe = self.modelframe.sort_values('best_score',ascending=False)
                df = self.modelframe
                model = str(df.iloc[0][0])
                param = str(df.iloc[0][2])
                param_str = str()
                for i,j in eval(str(param)).items():
                    tempL = str(i)
                    tempE = "="
                    if str(j).isnumeric():
                        tempR =f'int({j})'
                    elif self.isfloat(j):
                        tempR =f'float({j})'
                    else:
                        tempR = f"'{str(j)}'"
                    tempT = ","
                    param_str += (tempL+tempE+tempR+tempT)
                finalmodel = self.__modelCreater(model,param_str )
                logging.info(finalmodel.__class__.__name__)
                
            elif(model is not None and param is not None):
                model = str(model)
                param_str = str()
                for i,j in eval(str(param)).items():
                    param_str += (str(i)+"="+ str(j)+",")
                finalmodel = self.__modelCreater(model,param_str )
            else:
                raise Exception("model or param one of them is None")
                logging.debug(finalmodel.__class__.__name__)
        except Exception as e:
            logging.critical(str(e)+traceback.format_exc())
        else:
            logging.info(finalmodel.__class__.__name__)
            finalmodel.fit(X,y)
            logging.info('final fitting completed successfully')
            return finalmodel
    
        
    
    
class InferentialCore:
    '''
    Special class to answer infrential question
    it provide both linear regression & logistic regression
    '''
    def __init__(self,X,y,regu=0,l1regu=0,tune = False):
        '''
        Note:(OLS object).summary() will give INFRENTAIL ANALYSIS


        Parameters
        ----------
        X : TYPE Dataframe 
            DESCRIPTION. X
        y : TYPE Dataframe
            DESCRIPTION. y
        regu : TYPE int , optional
            DESCRIPTION.alpha regularization The default is 0.
        l1regu : TYPE int , optional
            DESCRIPTION.L1 regularization The default is 0.
        tune : TYPE Boolean , optional
            DESCRIPTION. Tune hyper paramter or notThe default is False.

        Returns
        -------
        None.

        '''
        self.X =X
        self.y = y
        self.X_train ,self.X_test,self.y_train ,self.y_test = train_test_split(self.X, self.y, test_size=0.33)
        self.regu = regu
        self.l1reg =l1regu
        self.tune = tune
        self.final_score =None
        self.metric = None
        logging.info('__datasetcreater completed successfully')
        
    def __mylogconfig(self):
        '''
        configure logging module

        Returns
        -------
        None.

        '''
        logging.basicConfig(filename="Inferential.log",level=logging.INFO,format="[ %(levelname)s :  %(asctime)s :"+__name__+" : %(funcName)17s() -%(lineno)d ] %(message)s")
        logging.info("Logging new SESSION")    
        
    def startRegression(self,metrics = r2_score,greater_isbetter = True):
        '''
        
        Regression for  INFRENTAIL ANALYSIS


        Parameters
        ----------
         metric : TYPE Sklearn.metric, optional
            DESCRIPTION. The default is accuracy_score.
        greater_isbetter : TYPE boolean, optional
            DESCRIPTION. The default is True.
        Returns
        -------
        result : TYPE OLS object
            DESCRIPTION. OLS object

        '''
        metrics = r2_score
        self.metrics = r2_score
        try:
            self.X = sm.add_constant(self.X,has_constant='add')
            if(self.tune):
                model = sm.OLS(self.y,self.X)
                result = model.fit()
                self.model = model
                print(1)
                return (self.model,result)
            else:
                model = sm.OLS(self.y_train,self.X_train)
                score = []
                if self.regu == 0 and self.l1reg ==0:
                    halpha = [0.001,0.01,0.1,1,5,10,0.5]
                    hL1_wt = [0.001,0.01,0.1,1,5,10,0.5]
                    for i in halpha:
                        for j in hL1_wt:
                                result =model.fit_regularized(alpha=i,L1_wt=j,refit = True)
                                score.append([self.metrics(self.y_test,result.predict(self.X_test)),i,j])
                    score.sort(key=lambda x:x[0],reverse=greater_isbetter)
                    model = sm.OLS(self.y,self.X)
                    result = model.fit_regularized(alpha=score[0][1],L1_wt=score[0][2],refit = True)
                    print(2)
                    self.model = model
                    return (self.model,result)
                else:
                    model = sm.OLS(self.y,self.X)
                    result =model.fit_regularized(alpha=self.regu,L1_wt=self.l1reg,refit = True)
                    self.model = model
                    print(3)
                    return (self.model,result)
        except Exception as e:
            print(traceback.format_exc())
            logging.critical(e)
            
            print(e)
            print(4)
        logging.info('startRegression completed successfully')
        
    def tosave(self):
        '''
        Save model on to drive

        Returns
        -------
        None.

        '''
        try:
            filename = 'finalized_model'+ self.id +'.sav'
            pickle.dump(self.model, open(filename, 'wb'))
        except Exception as e:
            print('file type error'+e)
        logging.info('tosave completed successfully')
        
    def toload(self,filename):
        '''
        Load model from disk

        Parameters
        ----------
        filename : TYPE String
            DESCRIPTION. name of file conatining model

        Returns
        -------
        None.

        '''
        try:
            loaded_model = pickle.load(open(filename, 'rb'))
            self.model = loaded_model
        except Exception as e:
            print('file type error'+e)
        logging.info('toload completed successfully')   
    def predict(self,model):
        '''
    
        if model is created and only predition are moade

        Parameters
        ----------
        model : TYPE sklearn model
            DESCRIPTION. sklean supervisemodel


        Returns
        -------
        temp : TYPE tuple(X featuer,y predication)
            DESCRIPTION. feature and predication

        '''
        try:
            temp =  (self.X,model.predict(model))
        except Exception as e:
            logging.critical(e)
        logging.info('def predict completed successfully')
        return temp
   
    def startLogistic(self,metrics = accuracy_score, greater_isbetter = True):
        '''
        Logistic REGRESSION FOR BINARY CLASSIFICATION CLASS MUST BE IN FORM OF 0/1 FOR INFRENTAIL ANALYSIS

        Parameters
        ----------
        metric : TYPE Sklearn.metric, optional
            DESCRIPTION. The default is accuracy_score.
        greater_isbetter : TYPE boolean, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        result : TYPE OLS object
            DESCRIPTION.OLS object

        '''
        metrics = accuracy_score
        self.metrics = accuracy_score
        try:
            self.X = sm.add_constant(self.X,has_constant='add')
            if(self.tune):
                self.model = sm.Logit(self.y,self.X)
                result= self.model.fit()
                return (self.model,result)
            else:
                model = sm.Logit(self.y_train,self.X_train)
                score = []
                if self.regu == 0 and self.l1reg ==0:
                    halpha = [0.001,0.01,0.1,1,5,10,0.5]
                    hL1_wt = [0.001,0.01,0.1,1,5,10,0.5]
                    for i in halpha:
                        for j in hL1_wt:
                                result =model.fit_regularized(alpha=i,L1_wt=j,refit = True)
                                score.append([self.metrics(self.y_test,list(map(round, result.predict(self.X_test)))),i,j])
                    score.sort(key=lambda x:x[0],reverse=greater_isbetter)
                    model = sm.Logit(self.y,self.X)
                    result = model.fit_regularized(alpha=score[0][1],L1_wt=score[0][2],refit = True)
                    self.model = model
                    return (self.model,result)
                else:
                    model = sm.Logit(self.y,self.X)
                    result =model.fit_regularized(alpha=self.regu,L1_wt=self.l1reg,refit = True)
                    self.model = model
                    return (self.model,result)
        except Exception as e:
            logging.critical(e)
            print(traceback.format_exc())
        logging.info('startLogistic completed successfully')
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#     if i in exp:
#         regressor_dict[i]={"max_iter":[3]}
#     elif i in exp1:
#         regressor_dict[i] = {"max_features":[1]}
#     elif i !="AdaBoostRegressor":
#         regressor_dict[i]={"n_jobs":[1]}
#     else:
#         regressor_dict[i] ={"n_estimators":[50]}
# classifier_dict  = {}
# for j in classifier:
#     classifier_dict[j] ={"n_jobs":[None]}
# import pandas as pd
# x = pd.read_csv(r"C:\Users\nemam\Desktop\ALL_COMBINE_DATASET\JEET\train.csv").iloc[:,0:-1]
# y = pd.read_csv(r"C:\Users\nemam\Desktop\ALL_COMBINE_DATASET\JEET\train.csv").iloc[:,-1]


# sv = SVLearningCore(X,y,,None,False,"high")
# cu = (sv.startlearning())

# =============================================================================
# sv.train_get(X,y)
# cu.to_csv(r"C:\Users\nemam\Untitled Folder\Major\df.csv")
# y=pd.DataFrame(y).replace({2:0})
# X_train,X_test,y_train,y_test = train_test_split(sns.load_dataset('iris')[["sepal_length" ,"sepal_width","petal_length"]],sns.load_dataset('iris')["petal_width"],test_size=0.2)
# re = InferentialCore(X,y,X_train,y_train,X_test,y_test,regu=0,l1regu=0,tune=True)
# res = re.startLogistic()
# print(res.summary())
# =============================================================================
