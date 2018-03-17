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
        pass
    def BestParam(self):
        Data = DataManager("Opioids", "../public_data")
        X = Data.data['X_train']
        Y = Data.data['Y_train']
        parametresDeTest = { 'n_estimators' : [10,50,100,200],
                            'max_features': ['auto', 'sqrt', 'log2'],
                            'bootstrap': [True, False]}
        classifier = GridSearchCV(RandomForestClassifier(), param_grid = parametresDeTest)
        classifier.fit(X, Y)
        
        nEstimatorOptimal = classifier.best_params_['n_estimators']
        maxFeaturesOptimal = classifier.best_params_['max_features']
        bootstrapOptimal = classifier.best_params_['bootstrap']
        
        print "nEstimators = {}".format(nEstimatorOptimal)
        print "Features = {}".format(maxFeaturesOptimal)
        print "Bootstrap = {}".format(bootstrapOptimal)
        
        return [nEstimatorOptimal, maxFeaturesOptimal, bootstrapOptimal]