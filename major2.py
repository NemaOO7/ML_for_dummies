# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 19:53:24 2022

@author: nemam
"""
from  sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import logging
import uuid,pickle
class USLearningCore:
    "Provide technique to assign labeled to unlabeled data"
    def __init__(self,X,n_clusters=1,max_cluter =11,max_itr = 300,n_init = 10):
        '''
        Unsupervised learning Constuctor,Conatin two type of algo Kmean and AgglomerativeClustering

        Parameters
        ----------
        X : TYPE Datafroame
            DESCRIPTION.Input data frame
        n_clusters : TYPE int, optional
            DESCRIPTION.number of cluster.if None algorithm will help to find them The default is None.
        max_cluter : TYPE int, optional
            DESCRIPTION.maximum number of possible cluster The default is 11.
        max_itr : TYPE int , optional
            DESCRIPTION. max number of iteration for convergence, The default is 300.
        n_init : TYPE, optional
            DESCRIPTION.different number of inialial value, The default is 10.

        Returns
        -------
        None.

        '''
        self.X = X
        self.n_clusters = n_clusters
        self.max_itr = max_itr
        self.n_init = n_init
        self.max_cluter = max_cluter
        self.id = uuid.uuid1
        self.__mylogconfig()
        logging.info('construtor completed successfully')
        
    def __mylogconfig(self):
        '''
        configure logging module

        Returns
        -------
        None.

        '''
        logging.basicConfig(filename="USLearningCore.log",level=logging.INFO,format="[ %(levelname)s :  %(asctime)s :"+__name__+" : %(funcName)17s() -%(lineno)d ] %(message)s")
        logging.info("Logging new SESSION")    
        
    def tosave(self,model):
        '''
        Save model on to drive
        Parameters
        ----------
        model : TYPE sklearn unsupervise model
            DESCRIPTION.model
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
        
    def predict(self,model):
        '''
    
        if model is created and only predition are moade

        Parameters
        ----------
        model : TYPE sklearn model
            DESCRIPTION. sklean unsupervisemodel


        Returns
        -------
        temp : TYPE tuple(X featuer,y predication)
            DESCRIPTION. feature and predication

        '''
        try:
            temp =  (self.X,model.transform(self.X))
        except Exception as e:
            logging.critical("either model or data are incorrent "+e)
        logging.info('def predict completed successfully')
        return temp
    
    def __find_clusters_elbow(self):
        '''
        function help to find cluter with the help of elbow mehod see were elbow is forming that is number of cluster

        Returns
        -------
        n_cluster : TYPE int 
            DESCRIPTION. number of cluster

        '''
        try:
            wcss = []
            for i in range(1,self.max_cluter):
                kmeans = KMeans(n_clusters = i,init = 'k-means++',max_iter=self.max_itr,n_init=self.n_init)
                kmeans.fit(self.X)
                wcss.append(kmeans.inertia_)
            plt.plot(range(1,self.max_cluter),wcss)
            plt.title('The Elbow Method ')
            plt.xlabel('Number of clusters')
            plt.ylabel('WCSS')
            #plt.show(block=False)
            plt.savefig("myImagePDF.jpg")
        except Exception as e:
            logging.critical("Somthing went wrong"+e)
        try:
            '''
                give number of cluster on the basis elbow given by user
            '''
            n_cluster =  self.n_clusters
        except Exception as e:
               logging.error("wrong input "+e)
        logging.info('__find_clusters_elbow completed successfully')
        return n_cluster
    
    def kmeanslearning(self):
        '''
        use Kmeans++ algorithm help to assign class to the given data

        Returns
        -------
        kmeans : TYPE sklearn Object
            DESCRIPTION. kmeans object
        y_kmeans : TYPE dataframe/array
            DESCRIPTION. give labels for corrosponding object

        '''
        try:
            if self.n_clusters == 1:
                n_cluster = self.__find_clusters_elbow()
            else:
                n_cluster = self.n_clusters
            kmeans = KMeans(n_clusters = n_cluster,init = 'k-means++',max_iter=self.max_itr,n_init=self.n_init)
            y_kmeans = kmeans.fit_predict(self.X)
        except Exception as e:
            logging.critical("Somthing went wrong"+e)
        logging.info('kmeanslearning completed successfully')
        return kmeans
    
    
    def __find_clusters_dendrogram(self):
        '''
        function help to find cluter with the help of dendrogram mehod see were longest uncut line is there horizantily

        Returns
        -------
        n_cluster : TYPE int
            DESCRIPTION. number of cluster


        '''
        try:
            dendrogram = sch.dendrogram(sch.linkage(self.X,method='ward'))
            plt.title('dendrogram')
            plt.xlabel('dataset')
            plt.ylabel('Eculidean distance')
            #plt.show(block=False)
            plt.savefig("myImagePDF.jpg")
            try:
                '''
                    give number of cluster on the basis elbow given by user
                '''
                n_cluster = self.n_clusters
            except Exception as e:
               logging.error("wrong input "+e)
        except Exception as e:
            logging.critical("Somthing went wrong"+e)
        logging.info('__find_clusters_dendrogram completed successfully')
        return n_cluster
    
    def agglomerativeClustering(self):
        '''
        use agglomerativeClustering algorithm help to assign class to the given data

        Returns
        -------
        y_pred : TYPE
            DESCRIPTION. classes assigned to y.

        '''
        try:
            if self.n_clusters == 1:
                n_cluster = self.__find_clusters_dendrogram()
            else:
                n_cluster = self.n_clusters
            hc =  AgglomerativeClustering(n_clusters=n_cluster,affinity="euclidean",linkage='ward')
            y_pred = hc.fit_predict(self.X)
        except Exception as e:
            logging.critical("Somthing went wrong"+e)
        logging.info('agglomerativeClustering completed successfully')
        return hc
def fun():
    import seaborn as sns
    df = sns.load_dataset('iris')[['sepal_length',  'sepal_width',  'petal_length' ,'petal_width' ]]
    n  = USLearningCore(df)
    n.agglomerativeClustering()

if __name__ == '__main__':
    fun()