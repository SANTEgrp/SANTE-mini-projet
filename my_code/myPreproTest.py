# -*- coding: utf-8 -*-
"""
Éditeur de Spyder
Ceci est un script temporaire.
"""

from data_manager import DataManager # such as DataManager
from myPrepro import Preprocessor 

#Récupération des fichiers nécessaires
input_dir = "../public_data"
output_dir = "../sample_results_submission"
    

basename = 'Opioids' 

Data = DataManager(basename, input_dir) # Création des données

'''Affichage des donnees originales'''

print("Data non modifiées")
print Data
    
#Affichage des données avant modification
import matplotlib.pyplot as plt
X = Data.data['X_train']
Y = Data.data['Y_train']
plt.scatter(X[:,0],X[:,1], c=Y)
plt.xlabel('DonneesX')
plt.ylabel('DonneesY')
plt.show()

'''Initialisation du Preprocessing'''

Prepro = Preprocessor() #Initialisation

'''Modification des donnees'''

#Preprocessing sur les données
Data.data['X_train'] = Prepro.fit_transform(Data.data['X_train'], Data.data['Y_train'])

Data.data['X_valid'] = Prepro.transform(Data.data['X_valid'])

Data.data['X_test'] = Prepro.transform(Data.data['X_test'])
  
'''Affichage des donnees modifiees'''

#Affichage des données modifiées + petit graphique pour que ce soit visible
print("Data modifiées après le Préprocessing")
print Data

import matplotlib.pyplot as plt
X = Data.data['X_train']
Y = Data.data['Y_train']
plt.scatter(X[:,0],X[:,1], c=Y)
plt.xlabel('SGCDClassifier1')
plt.ylabel('SGCDClassifier2')
plt.show()