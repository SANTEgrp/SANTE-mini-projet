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
from MyBestParametres import BestParametres
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

if __name__=="__main__":
    input_dir="../public_data"
    output_dir="../sample_result_submission"      
    basename = "Opioids"
    
    from data_manager import DataManager
    
    D = DataManager(basename, input_dir) #Initialisation des données utilisées par le classifieur
    print D #Affichage de test pour voir les données prises sont les bonnes
    
    Classifier = model() #Initialisation du classifieur (ici il prend celui de model)
    
    XTrain_data = D.data['X_train'] #Données d'entrainement
    YTrain_data = D.data['Y_train'] #Donnés de test
    fit = Classifier.fit(XTrain_data, YTrain_data) 
    fit #Fit des données d'entrainement et  des données cibles
    
    YTrainPredict = Classifier.predictProba(D.data['X_train'])
    
    YValidPredict = Classifier.predictProba(D.data['X_valid'])
    
    YTestPredict = Classifier.predictProba(D.data['X_test'])
    #Création des prédictions sur les données pour calculer les résultats du Classifieur
    
    from my_metric import auc_metric_
    
    classifieurAuc = auc_metric_(YTrain_data, YTrainPredict)
    print "\n"
    print "Score du classifieur: "
    print classifieurAuc
    print "\n"
    #A noter que l'on obtient un score à 0.99 ce qui  semble étrange TODO: A Corriger
    
    from sklearn.model_selection import cross_val_score
    
    print "Cross Validation: "
    clf = RandomForestClassifier(n_estimators = 100, max_features = "log2", bootstrap = True)
    crossval = cross_val_score(clf, XTrain_data, YTrain_data, cv = 3) #Calcule de la cross validation, 3 fois
    print crossval #Affichage des 3 cross validations
    print("Precision: %0.4f (+/- %0.04f)" % (crossval.mean(), crossval.std() * 2)) #Affichage de la moyenne des 3 CV +la précision
    print "\n"
    
    from sklearn.metrics import confusion_matrix
    
    print "Matrice de confusion: "
    print confusion_matrix(YTrain_data, Classifier.predict(D.data['X_train']))
    
    
    
    
    
    
    
    
    
    
        
        
        
