# -*- coding: utf-8 -*-
"""
Le preprocessing
"""


from sklearn.base import BaseEstimator

#Pour les fonctions de preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

class Preprocessor(BaseEstimator):

    def __init__(self):
        
        #Sélection de la fonction de preprocessing    
        model = SGDClassifier() #algorithme de descente du gradient stochastique
        #model = LinearSVC()  #2e modèle possible pour le preprocessing
        self.transformer = SelectFromModel(model)                
        
    def fit(self, X, y=None):

        return self.transformer.fit(X, y)


    def fit_transform(self, X, y=None):

        return self.transformer.fit_transform(X,y)


    def transform(self, X, y=None):

        return self.transformer.transform(X)
    
if __name__=="__main__":

    #Initialisation de la position des données
    input_dir = "../public_data"
    output_dir = "../sample_results_submission"
    basename = 'Opioids'

    from data_manager import DataManager 

    Data = DataManager(basename, input_dir) #Chargement des données
    
    #Affichage des données non retouchées
    print("Data Original")
    print Data     

    Prepro = Preprocessor() #J'appelle le preprocessing
 

    #On fait le preprocessing sur les données
    Data.data['X_train'] = Prepro.fit_transform(Data.data['X_train'], Data.data['Y_train'])
    Data.data['X_valid'] = Prepro.transform(Data.data['X_valid'])
    Data.data['X_test'] = Prepro.transform(Data.data['X_test'])
  
    #On vérifie si il y a des changements
    print("Data modifiées")
    print Data
    
    
    




