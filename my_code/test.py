# -*- coding: utf-8 -*-
"""
Éditeur de Spyder
Ceci est un script temporaire.
"""

from data_manager import DataManager # such as DataManager

from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import VarianceThreshold


class Preprocessor(BaseEstimator):

    def __init__(self):

        self.transformer = VarianceThreshold(threshold=(.8 * (1 - .8)))
    print("la valeur de self transformer est "+str(self.transformer)) 

    def fit(self, X, y=None):

        data= self.transformer.fit(X, y)
    #affichage de la valeur du data
    print ("la valeur de fit est"+ str(data))
    #la valeur retournée par la fonction fit
    return data 
  
  
    def fit_transform(self, X, y=None):

        return self.transformer.fit_transform(X,y)


    def transform(self, X, y=None):

        return self.transformer.transform(X)
    
if __name__=="__main__":

    input_dir = "../public_data"
    output_dir = "../sample_results_submission"
    

    basename = 'Opioids'

    Data = DataManager(basename, input_dir) # Load data

    print("*** Original data ***")

    print Data
    

    Prepro = Preprocessor()
 

    # Preprocess on the data and load it back into D

    Data.data['X_train'] = Prepro.fit_transform(Data.data['X_train'], Data.data['Y_train'])

    Data.data['X_valid'] = Prepro.transform(Data.data['X_valid'])

    Data.data['X_test'] = Prepro.transform(Data.data['X_test'])
  

    # Here show something that proves that the preprocessing worked fine

    print("*** Transformed data ***")
    print(__init__(5))
    print(fit(self, , None))
    print(fit_transform(self, X, None))
    print(transform(self, X, None))
    print Data
