# -*- coding: utf-8 -*-
"""
Éditeur de Spyder
Ceci est un script temporaire.
"""

from data_manager import DataManager # such as DataManager
from myPrepro import Preprocessor 




    


input_dir = "../public_data"
output_dir = "../sample_results_submission"
    

basename = 'Opioids'

Data = DataManager(basename, input_dir) # Load data

print("Data non modifiées")
print Data
    

Prepro = Preprocessor()
import matplotlib.pyplot as plt
X = Data.data['X_train']
Y = Data.data['Y_train']
plt.scatter(X[:,0],X[:,1], c=Y)
plt.xlabel('DonneesX')
plt.ylabel('DonneesY')
plt.show()


    # Preprocess on the data and load it back into D

Data.data['X_train'] = Prepro.fit_transform(Data.data['X_train'], Data.data['Y_train'])

Data.data['X_valid'] = Prepro.transform(Data.data['X_valid'])

Data.data['X_test'] = Prepro.transform(Data.data['X_test'])
  

    # Here show something that proves that the preprocessing worked fine

print("Data modifiées après le Préprocessing")
print Data

import matplotlib.pyplot as plt
X = Data.data['X_train']
Y = Data.data['Y_train']
plt.scatter(X[:,0],X[:,1], c=Y)
plt.xlabel('SelectKBest1')
plt.ylabel('SelectKBest2')
plt.show()