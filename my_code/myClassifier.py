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
from sklearn.ensemble import RandomForestClassifier
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
        
        algo = RandomForestClassifier(n_estimators = 100, max_features = "sqrt", bootstrap = False)
        #Ces parametres ont ete trouves en exécutant en local BestParametres.py

        self.mymodel = Pipeline([('preprocessing',Preprocessor()),('class',algo)])
        #Application du preprocessing
        
    '''
    Construit une forêt d'arbre selon les datas (X,y)
    '''
    def fit(self, X, y):
       #Xtr = self.mymodel.fit_transform(X, y)
       return self.mymodel.fit(X, y)

    '''
    Retourne la probabilité d'une donnée X d'être dans la classe 1
    '''
    def predict_proba(self, X):
  
        return self.mymodel.predict_proba(X)[:,1]
    
    '''
    Retourne la classe prédite d'une donnée. Ici, vu qu'on est dans un problème de
    classification binaire on préférera utiliser predict_proba
    '''
    def predict(self, X):
        return self.mymodel.predict(X)
        
    
    '''
    Sauvegarde le modèle crée
    '''
    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))
    '''
    Charge un modèle si il est présent
    '''
    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile) as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self

'''Exécuté que si on exécute la classe
'''
if __name__=="__main__":
    input_dir="../public_data"
    output_dir="../sample_result_submission"      
    basename = "Opioids"
    
    from data_manager import DataManager
    
    
    
    Classifier = model() #Initialisation du classifieur (ici il prend celui de model)
    
    testParam = 0 #Si on a besoin de trouver les paramètres optimaux
    if testParam == 1:
            #Ces trois initialisations sont utlisées quand on ne soumet pas car Codalab
            #ne prends pas en charge le "MyBestParametres.py"
            #Le self.mymodel permet de choisir les meilleurs paramètres fournis par myBestParamètres.py
            #qui retourne un tableau de taille 3"
            from MyParametresOpti import BestParametres
            params = BestParametres()
            estimators = params.BestParam()
            print "------------------------Meilleurs paramètres-------------------------"
            print "NEstimators: " + str(estimators[0])
            print "Max_features: " + str(estimators[1])
            print "Bootsrap: " + str(estimators[2])
            print "--------------------------------------------------------------"
            #algo = RandomForestClassifier(n_estimators = estimators[0], max_features = estimators[1], bootstrap = estimators[2])
            #Ce dernier récupère les trois paramètres donnés par "MyBestParametres.py"
   
    
    
    D = DataManager(basename, input_dir) #Initialisation des données utilisées par le classifieur
    print D #Affichage de test pour voir les données prises sont les bonnes
    
    XTrain_data = D.data['X_train'] #Données d'entrainement
    YTrain_data = D.data['Y_train'] #Donnés de test
    fit = Classifier.fit(XTrain_data, YTrain_data)     #Fit des données d'entrainement et  des données cibles


    need_print_fit = 0
    if need_print_fit == 1:
        print (fit)
    
    #Classifier.mymodel.fit_transform(XTrain_data, YTrain_data)
    
    
    YTrainPredict = Classifier.predict_proba(D.data['X_train'])
    
    YValidPredict = Classifier.predict_proba(D.data['X_valid'])
    
    YTestPredict = Classifier.predict_proba(D.data['X_test'])
    #Création des prédictions sur les données pour calculer les résultats du Classifieur
    
    from my_metric import auc_metric_
    
    classifieurAuc = auc_metric_(YTrain_data, YTrainPredict)
    print "\n"
    print "Score du classifieur: "
    print classifieurAuc
    print "\n"
    
    from sklearn.model_selection import cross_val_score
    
    print "Cross Validation: "
    clf = RandomForestClassifier(n_estimators = 100, max_features = "sqrt", bootstrap = False)
    crossval = cross_val_score(clf, XTrain_data, YTrain_data, cv = 3, scoring = 'roc_auc') #Calcule de la cross validation, 3 fois
    print crossval #Affichage des 3 cross validations
    print("Precision: %0.4f (+/- %0.04f)" % (crossval.mean(), crossval.std() * 2)) #Affichage de la moyenne des 3 CV +la précision
    print "\n"
    
    from sklearn.metrics import confusion_matrix
    
    print "Matrice de confusion: "
    print confusion_matrix(YTrain_data, Classifier.predict(D.data['X_train']))
    
    
    
    
    
    
    
    
    
    
        
