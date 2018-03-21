# -*- coding: utf-8 -*-

"""
Calcule le meilleur parametre pour la randomForest"
"""

from sys import path
path.append("../scoring_program")
path.append("../ingestion_program")
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from data_manager import DataManager

class BestParametres(GridSearchCV):
    def __init__(self):
        pass #Ici pas de constructeurs
    def BestParam(self):
        Data = DataManager("Opioids", "../public_data")
        X = Data.data['X_train'] #Data
        Y = Data.data['Y_train'] #Cible
        
        parametresDeTest = { 'n_estimators' : [2,5,8,10,50,100],
                            'max_features': ['auto', 'sqrt', 'log2'],
                            'bootstrap': [True, False]}#Les différents paramètres à tester
        
        classifier = GridSearchCV(RandomForestClassifier(), param_grid = parametresDeTest) #Récupère les résultats de la CrossValidation
        classifier.fit(X, Y) #Fit les résultats
        print "Meilleurs paramètres: "
        print (classifier.best_params_) #Affiche les paramètres qui fournissent la plus haute cross validation
        print "\n"
         
        
        return [classifier.best_params_['n_estimators'],classifier.best_params_['max_features'],classifier.best_params_['bootstrap']]
        #Retour des meilleurs paramètres dans un tableau pour y accéder plus facilement depuis le classifieur