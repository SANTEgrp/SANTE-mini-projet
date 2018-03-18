# -*- coding: utf-8 -*-
'''
Sample predictive model.
You must supply at least 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
- save: saves the model.
- load: reloads the model.
'''
import pickle
import numpy as np   # We recommend to use numpy arrays
from sys import argv
from os.path import isfile
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
#from MyBestParametres import BestParametres
from myPrepro import Preprocessor
from sklearn.pipeline import Pipeline

class model:
    def __init__(self):
        '''
        Initialisation des constructeurs
        '''
        self.num_train_samples=0
        self.num_feat=1
        self.num_labels=1
        self.is_trained=False
        
        '''
        Ces trois initialisations sont utlisées quand on ne soumet pas car Codalab
        ne prends pas en charge le "MyBestParametres.py"
        Le self.mymodel permet de choisir les meilleurs paramètres fournis par myBestParamètres.py
        qui retourne un tableau de taille 3"
        '''
        #self.params = BestParametres()
        #self.estimators = self.params.BestParam()
        #algo = RandomForestClassifier(n_estimators = self.estimators[0], max_features = self.estimators[1], bootstrap = self.estimators[2])
        #Ce dernier récupère les trois paramètres donnés par "MyBestParametres.py"
        
        
        
        
        algo = RandomForestClassifier(n_estimators = 200, max_features = "log2", bootstrap = True)
        #Ces parametres ont ete trouves en exécutant en local BestParametres.py

        self.mymodel = Pipeline([('preprocessing',Preprocessor()),('class',algo)])
        #Application du preprocessing
        
    def fit(self, X, y):
       
       return self.mymodel.fit(X, y)

    def predictProba(self, X):
  
        return self.mymodel.predict_proba(X)[:,1]
    
    def predict(self, X):
        return self.mymodel.predict(X)
    
    

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile) as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self
