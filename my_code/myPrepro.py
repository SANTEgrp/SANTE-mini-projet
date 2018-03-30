# -*- coding: utf-8 -*-
"""
Le preprocessing
"""


from sklearn.base import BaseEstimator

#Pour les fonctions de preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import GenericUnivariateSelect

class Preprocessor(BaseEstimator):

    def __init__(self):
        '''
        Initialisation de la fonction de preprocessing
        '''
        
        self.transformer = GenericUnivariateSelect(mode ='fwe', param = 1e-01)
                
        
    def fit(self, X, y=None):
        '''Centrage des donnees'''

        return self.transformer.fit(X, y)


    def fit_transform(self, X, y=None):
        '''Appel de fit et transform pour le set d'entrainement'''

        return self.transformer.fit_transform(X,y)


    def transform(self, X, y=None):
        '''Application de la transformation'''

        return self.transformer.transform(X)
    

    '''Execution si on est dans le main'''    
    
if __name__=="__main__":

    #Initialisation de la position des données
    input_dir = "../public_data"
    output_dir = "../sample_results_submission"
    basename = 'Opioids'

    from data_manager import DataManager 

    Data = DataManager(basename, input_dir) #Chargement des données
    
    '''On affiche les donnees originales'''
    #Affichage des données non retouchées
    print("Data Original")
    print Data     

    '''Appel de la classe Preprocessor'''
    Prepro = Preprocessor() #J'appelle le preprocessing
 

    '''Transformation des donnees'''
    #On fait le preprocessing sur les données
    Data.data['X_train'] = Prepro.fit_transform(Data.data['X_train'], Data.data['Y_train'])
    Data.data['X_valid'] = Prepro.transform(Data.data['X_valid'])
    Data.data['X_test'] = Prepro.transform(Data.data['X_test'])
    
    '''On affiche les donnees modifiees pour comparer et voir les changements'''
    #On vérifie si il y a des changements
    print("Data modifiées")
    print Data
    
    
    




