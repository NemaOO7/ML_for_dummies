# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 20:58:17 2022

@author: nemam
"""
from sklearn.decomposition import  PCA,KernelPCA
from sklearn.discriminant_analysis import  LinearDiscriminantAnalysis
import logging
import uuid,pickle
class DReductionCore:
    '''
    Usage a various dimensionality reduction technique to reduce dimension of data
    the class include pca,kernalPCA,LinearDiscriminantAnalysis
    '''
    def __init__(self,X,dim=2):
        '''
        

        Parameters
        ----------
        X : TYPE Dataframe
            DESCRIPTION.
        dim : TYPE int, optional
            DESCRIPTION.how to dimension want to reduce The default is 2.

        Returns
        -------
        None.

        '''
        self.X = X
        self.dim = min(dim,len(X.columns))
        self.__mylogconfig()
        self.id = uuid.uuid1
        logging.info('DReductionCore construtor completed successfully')
    
    def tosave(self,model):
        '''
        Save model on to drive

        Returns
        -------
        None.

        '''
        try:
            filename = 'finalized_model'+ self.id +'.sav'
            pickle.dump(model, open(filename, 'wb'))
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
    def reduce(self,model):
        '''
    
        if model is created and only predition are made

        Parameters
        ----------
        model : TYPE sklearn model
            DESCRIPTION. sklean reducation


        Returns
        -------
        temp : TYPE tuple(X featuer,y predication)
            DESCRIPTION. feature and predication

        '''
        try:
            temp =  (self.X,model.transform(self.X))
        except Exception as e:
            logging.critical("either model or data are incorrent "+e)
        logging.info('def reduce completed successfully')
        return temp
    
    def __mylogconfig(self):
        '''
        configure logging module

        Returns
        -------
        None.

        '''
        logging.basicConfig(filename="DReductionCore.log",level=logging.INFO,format="[ %(levelname)s :  %(asctime)s :"+__name__+" : %(funcName)17s() -%(lineno)d ] %(message)s")
        logging.info("Logging new SESSION")    
    
    def pcaReducation(self):
        '''
        pca readucation for linear data  it reduce dimension and try to preserve as muchd data as possible

        Returns
        -------
        pca : TYPE PCA object
            DESCRIPTION. pca model
        y_pca : TYPE array (dim = self.dim)
            DESCRIPTION. reduced dimension
        explained_variance : TYPE array
            DESCRIPTION.how much information is retained

        '''
        try:
            pca = PCA(n_components=self.dim)
            pca.fit(self.X,)
            y_pca = pca.transform(self.X)
            explained_variance = pca.explained_variance_
        except Exception as e:
            logging.critical("Somthing went wrong "+e)
        logging.info('pcaReducation completed successfully')
        return pca
    
    def kernalpcaReducation(self):
        '''
        pca readucation for non linear data, it reduce dimension and try to preserve as muchd data as possible

       Returns
        -------
        pca : TYPE PCA object
            DESCRIPTION. pca model
        y_pca : TYPE array (dim = self.dim)
            DESCRIPTION. reduced dimension
        explained_variance : TYPE array
            DESCRIPTION.how much information is retained
        '''
        try:
            pca = KernelPCA(n_components=self.dim,)
            pca.fit(self.X)
            y_pca = pca.transform(self.X)
            return pca
        except Exception as e:
            logging.critical("Somthing went wrong "+e)
        logging.info('kernalpcaReducation completed successfully')
        
    def ldaseparation(self,y):
        '''
        lda is dimensionilty reduction techtechnique that focus on separating the data along with reduction in dimension

        Parameters
        ----------
        y : TYPE Target varoable
            DESCRIPTION. Target variable to feature

        Returns
        -------
        lda : TYPE LDA modeltrained Object
            DESCRIPTION. LDA modeltrained Object
        y_lda : TYPE array
            DESCRIPTION.separated value

        '''
        try:
            y=y.astype('int')
            lda = LinearDiscriminantAnalysis(n_components=self.dim)
            lda.fit(self.X,y)
            y_lda  = lda.transform(self.X)
            return lda
        except Exception as e:
            logging.critical("Somthing went wrong "+e)
        logging.info('ldaseparation completed successfully')

    
#import seaborn as sns
#df = sns.load_dataset('iris')[['sepal_length',  'sepal_width',  'petal_length' ]]
#n  = DReductionCore(df)
#n.ldaseparation( sns.load_dataset('iris')['petal_width' ])