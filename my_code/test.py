# -*- coding: utf-8 -*-
"""
Éditeur de Spyder
Ceci est un script temporaire.
"""

from data_manager import DataManager # such as DataManager
from Prepro import Preprocessor 


class Preprocessor(BaseEstimator):

    
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
