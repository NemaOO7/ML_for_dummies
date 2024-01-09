# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 13:09:10 2022

@author: sabhi
"""
import numpy as np
import pandas as pn
#import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer
import sys
class dataprocessing:
    """
    The class is used to preprocess data 
    ...
    
    Attributes:
    -----------    
        path1 : str
               A string that holds path of first dataset to be uploaded
        path2: str
               A string that holds path of second dataset to be uploaded ,if any
    Methods:
    -------- 
        merge_dataset(): To merge two different dataset
        cleaningNA(): To remove the columns that have all values as None
        cleaningNAChoice(): To remove the columns that have any values as None
        categoricalcols(): To get the columns with all string values
        filling_missingdata(): To fill the missing data in columns with integer values
        filling_missingdatastring(): To fill the missing data in the columns with string values
        onehotencoding(ncol): To encode the string valued columns into integer
        removeoutliner(self,ncol=None): To remove the extreme valued data
        masterfunc(): To call all function together
        summary(): To present summary detail of dataset
        alertNa(): To find any colmuns with null values
        xconfig(xcols): To select colmuns as independent variable x
    """
    def __init__(self,path1,path2=None):
        """
        
        Parameters
        ----------
        path1 : str
            path of first Dataset.
        path2 : str,optional
            Path of second Dataset, if any. 
        
        Returns
        -------
        None.

        """
        self.path1 = path1
        if path2==None:
            self.path=path1
            pth=list()
            pth=path1.split("\\")
            try:
                self.filename=pth[-1]                 #expection handling for too large file  
                self.dataset=pn.read_csv("%s"%(self.path)) #read csv
            
            except:
                print(sys.exc_info()[0], "occurred.")
               
        else:
            self.path=path1
            pth=list()
            pth=path1.split("\\")
            try:    
                self.filename=pth[-1]                 #expection handling for too large file  
                self.dataset=pn.read_csv("%s"%(self.path)) #read csv and store it in self.dataset
            except:
                print(sys.exc_info()[0], "occurred.")
                print(sys.exc_info()[1], "occurred.")
            self.path1=path2
            pth1=list()
            pth1=path2.split("\\")
            try:
                self.filename1=pth1[-1]                 #expection handling for too large file  
                self.dataset1=pn.read_csv("%s"%(self.path1)) #read csv
            except:
                print(sys.exc_info()[0], "occurred.")
                print(sys.exc_info()[1], "occurred.")
    def merge_dataset(self):
        
        try:
            self.dataset=pn.concat([self.dataset, self.dataset1], axis=1, join="inner")
            #self.override(self.dataset, self.path1)
        except:
            print(sys.exc_info()[0], "occurred.")
            print(sys.exc_info()[1], "occurred.")
        
    def alertNa(self, path1):
        try:
            nan_cols = [i for i in self.dataset.columns if self.dataset[i].isnull().any()]
            print(nan_cols)
            with open('NULL.csv', 'w') as f:
                for item in nan_cols:
                    f.write("%s\n" % item)
            self.override(self.dataset, path1)
        except:
            print(sys.exc_info()[0], "occurred.")
            print(sys.exc_info()[1], "occurred.")
    
    def is_NA(self, path1):
        try:
            nan_cols = [i for i in self.dataset.columns if self.dataset[i].isnull().any()]
            print(nan_cols)
            if(nan_cols):
                return True
            else:
                return False
            self.override(self.dataset, path1)
        except:
            print(sys.exc_info()[0], "occurred.")
            print(sys.exc_info()[1], "occurred.")
            
    def xconfig(self,xcols, ncol, path1):
        try:            
            self.tcols=list(self.dataset.columns)
            print(self.tcols)
            xcols.append(ncol)
            print("THE LIST IS NOW ",xcols)
            for i in xcols:    
                if i in self.tcols:
                    self.tcols.remove(i)
                print(self.tcols)    
            self.dataset=self.dataset.drop(columns=self.tcols,axis=1)
            print(self.dataset)
            print("rebuilding the dataset")
            self.override(self.dataset, path1)
        except:
            print("THIS IS TH EXCEPTION IN XCONFIG")
            print(sys.exc_info()[0], "occurred.")
            print(sys.exc_info()[1], "occurred.")

    def get_x(self,xcols, path1):
        try:  
            temp = pn.read_csv(path1)[xcols] 
            print("$$$$$$$$",temp)         
            return temp
        except:
            print(sys.exc_info()[0], "occurred.")
            print(sys.exc_info()[1], "occurred.")
    
    def get_y(self,xcols, path1):
        try:
            temp = pn.read_csv(path1)[xcols]
            print("$$$$000000$$$$",temp)          
            return temp
           
        except:
            print(sys.exc_info()[0], "occurred.")
            print(sys.exc_info()[1], "occurred.")

    def cleaningNA(self, path1):
        try:
            self.dataset=self.dataset.dropna(axis=1,how='all')
            #self.dataset=self.dataset.loc[:,self.dataset.notna().any(axis=0)]
            self.dataset=self.dataset.dropna(axis=0,how='all')
            if (self.dataset.empty==True):
                raise Exception("Dataset is empty")
            self.override(self.dataset, path1)
        except:     
            print('1',sys.exc_info()[0], "occurred.")
            print('1',sys.exc_info()[1], "occurred.")
    def cleaningNAChoice(self, path1):
        try:
            #self.dataset=self.dataset.dropna(axis=1,how='any')
            self.dataset=self.dataset.dropna(axis=0,how='any')
            if (self.dataset.empty==True):
                raise Exception("Dataset is empty")
            self.override(self.dataset, self.path1)
        except:
            print('2',sys.exc_info()[0], "occurred.")
            print('2',sys.exc_info()[1], "occurred.")
    def categoricalcols(self, path1):
        self.categorical=self.dataset.select_dtypes(include='object') #to get columns with string data type
        self.col=list()
        self.col=self.categorical.columns # for the names of colmuns which have catergorical value
        self.override(self.dataset, path1)
    def filling_missingdata(self, path1):
        #self.dataset3=self.dataset
        imputer=SimpleImputer(missing_values=np.nan,strategy='median')  # for all columns except string type 
        try:
            imputer=imputer.fit(self.dataset.loc[:, ~self.dataset.columns.isin(self.col)])
            self.dataset.loc[:, ~self.dataset.columns.isin(self.col)]=imputer.transform(self.dataset.loc[:, ~self.dataset.columns.isin(self.col)])
            self.override(self.dataset, path1)
        except:
            print('3',sys.exc_info()[0], "occurred.")
            print('3',sys.exc_info()[1], "occurred.")
    def filling_missingdatastring(self, path1):    
        try:            
            imputer1=SimpleImputer(missing_values=np.nan,strategy='most_frequent') #for string type columns
            imputer1=imputer1.fit(self.dataset.loc[:, self.dataset.columns.isin(self.col)])
            self.dataset.loc[:,self.dataset.columns.isin(self.col)]=imputer1.transform(self.dataset.loc[:,self.dataset.columns.isin(self.col)])
            self.override(self.dataset, path1)
        except:    
            print('4',sys.exc_info()[0], "occurred.")
            print('4',sys.exc_info()[1], "occurred.")
            
    def summary(self):
        try:
            print(self.dataset.describe())
            print(type(self.dataset.describe()))
            self.dataset.describe().to_csv('insight.csv')
            #self.override(self.dataset)
        except:
            print('5',self.dataset,sys.exc_info()[0], "occurred.")
            print('5',self.dataset,sys.exc_info()[1], "occurred.")
    def labelencoding(self,ncol, path1):    
        #leb=preprocessing.LabelEncoder()
        try:
            ncol=LabelEncoder().fit_transform(ncol)
            print("@",ncol)
            #self.dataset[self.col]=self.dataset[self.col].apply((lambda col: le.fit_transform(col)))
            self.override(self.dataset, path1)
        except:
            print(sys.exc_info()[0], "occurred.")
            print(sys.exc_info()[1], "occurred.")
    def  onehotencoding(self,ncol, path1):
        """
        

        Parameters
        ----------
        ncol : str
            The name of column that is not to be encoded

        Returns
        -------
        None.

        """
        #column_trans = ColumnTransformer([("onehot_categorical", OneHotEncoder(sparse=False),self.col),],remainder="passthrough",)  # or drop if you don't want the non-categoricals at all...
        #one=OneHotEncoder(categories =self.categorical,sparse=False)
        #self.dataset4=one.fit_transform(self.dataset)
        #self.arr=column_trans.fit_transform(self.dataset)
        try:
            lit=list(self.col)
            if ncol in lit:    
                lit.remove(ncol)
                self.arr = pn.get_dummies(self.dataset[lit])
                #self.dataset4=pn.DataFrame(self.arr,columns=self.dataset3.columns)
                #self.dataset4=pn.DataFrame(self.arr)
                leb=[self.arr,self.dataset.loc[:, ~self.dataset.columns.isin(self.col)]]
                if len(leb)!=0:
                    self.dataset=pn.concat(leb,axis=1)
            self.override(self.dataset, path1)
        except:
            print(sys.exc_info()[0], "occurred.")
            print(sys.exc_info()[1], "occurred.")
        
    def removeoutliner(self,path1,ncol=None):
        """
        

        Parameters
        ----------
        ncol : str optional
            The column to be left aside. The default is None.

        Returns
        -------
        None.

        """
        try:    
            lower = self.dataset.quantile(.02)# for minimum extrema
            upper = self.dataset.quantile(.99)# for maximum extrema
            self.dataset=self.dataset.clip(lower=lower, upper=upper, axis=1)
            self.override(self.dataset, path1)
        except:
            print(sys.exc_info()[0], "occurred.")
            print(sys.exc_info()[1], "occurred.")
    def masterfunc(self, p1, y, x, n=1):
        p2=None
        ls=['Film','Rotten Tomatoes Ratings %','Budget (million $)','Audience Ratings %']  
        obj=dataprocessing(p1,p2)
        print("File name is ",p1)
        print("INITIATED")

        print("X is : ",x)  
        print("Y is : ",y)
        obj.xconfig(x)
        print("XCONFIG DONE")
        print("After xconfig ", )
        obj.alertNa()
        print("ALERTNA DONE")
        obj.cleaningNA()
        print("CLEANINGNA DONE")
        obj.removeoutliner()
        print("REMOVEOUTLIER DONE")
        obj.categoricalcols()
        print("CATEGORICALS DONE")
        obj.filling_missingdata()
        print("FILLING MISSING DATA DONE")
        obj.filling_missingdatastring()
        print("FILLING MISSING DATA STRING DONE")
        obj.labelencoding()
        print("LABEL ENCODING DONE")
        obj.onehotencoding(y)  
        print("ONE HOT ENCODING DONE")
        obj.summary()
        print("SUMMARY DONE DONE")
        self.override(obj.dataset, p1)
        print("The final data now is ")
        d = pn.read_csv("%s"%(p1))
        print(d)
    
    def override(self, df, fname):
        print("this is ")
        print(df.to_string())
        df.to_csv(fname, index=False, encoding='utf-8')
        print("The data now is ")
        print("fname is ",fname)
        d = pn.read_csv("%s"%(fname))
        print(d)

    # def main(p1, y, x, n=1):
    #     # n=int(input("Enter the no. of the dataset: "))        
    #     # p1=None  #C:\Users\sabhi\Downloads\P4-Movie-Ratings1.csv
    #     # p1=input("enter the path: ")
    #     p2=None
    #     # if n==2:
    #     #     #C:\Users\sabhi\Downloads\2022_3_14_23_45_9.csv
    #     #     p2=input("enter the path: ")
    #     #     obj=dataprocessing(p1,p2)
    #     #     obj.merge_dataset()
    #     # string_columns=list()
    #     # print("Main function is called")
    #     # obj=dataprocessing(p1,p2) 
    #     obj.categoricalcols()
    #     obj.xconfig(x)
    #     # obj.alertNa()
    #     # obj.cleaningNA()
    #     # #obj.removeoutliner()
    #     # obj.categoricalcols()
    #     # obj.filling_missingdata()
    #     # obj.filling_missingdatastring()
    #     # #obj.labelencoding()
    #     # obj.onehotencoding(y)  
    #     # obj.summary()
    #     # print("data cleaning completed")
    #     # #error handling
    #     # #log filling
    #     # #extremas value remove 
    #     # #documentation
      
# n=int(input("Enter the no. of the dataset: "))        
# p1=None  #C:\Users\sabhi\Downloads\P4-Movie-Ratings1.csv
# p1=input("enter the path: ")
# p2=None
# if n==2:
#      #C:\Users\sabhi\Downloads\2022_3_14_23_45_9.csv
#     p2=input("enter the path: ")
#     obj=dataprocessing(p1,p2)
#     obj.merge_dataset()
# ls=['Film','Rotten Tomatoes Ratings %','Budget (million $)','Audience Ratings %']  
# obj=dataprocessing(p1,p2)  
# obj.xconfig(ls)
# obj.alertNa()
# obj.cleaningNA()
# obj.removeoutliner()
# obj.categoricalcols()
# obj.filling_missingdata()
# obj.filling_missingdatastring()
# #obj.labelencoding()
# obj.onehotencoding('Audience Ratings %')  
# obj.summary()

# #error handling
# #log filling
# #extremas value remove 
# #documentation