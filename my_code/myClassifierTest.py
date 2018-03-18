#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""


@author: simon.rouff
"""

from data_manager import DataManager
from myClassifier import model
from my_metric import auc_metric_
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix



input_dir = "../public_data"
output_dir = "../sample_result_submission"

#Initialisation des données
Data = DataManager("Opioids", input_dir)
#Vérification que ce sont les bonnes données
print Data

#Initialisation du classifieur
Classifieur = model()

XTrainData = Data.data['X_train']
YTrainData = Data.data['Y_train']
Classifieur.fit(XTrainData,YTrainData)

#Predictions des probas:
XTrainPredictProba = Classifieur.predictProba(Data.data['X_train'])
XTestPredictProba = Classifieur.predictProba(Data.data['X_test'])
XValidPredictProba = Classifieur.predictProba(Data.data['X_valid'])

#Predictions: 
XTrainPredict = Classifieur.predict(Data.data['X_train'])
XTestPredict = Classifieur.predict(Data.data['X_test'])
XValidPredict = Classifieur.predict(Data.data['X_valid'])

#Score de l'AUC (notre propore métrique)
print"\n Score auc: %0.4f \n" % auc_metric_(YTrainData, XTrainPredict)

#Score Cross Validation:
clf = RandomForestClassifier(n_estimators = 100, max_features = "log2", bootstrap = True)
cvalid = cross_val_score(clf, XTrainData, YTrainData, cv = 5, scoring = "accuracy")
print("Cross-Validation: %5.2f (+/- %5.2f)" % (cvalid.mean(), cvalid.std() * 2))

print "Matrice de confusion: "
print confusion_matrix(YTrainData, Classifieur.predict(Data.data['X_train']))
 
