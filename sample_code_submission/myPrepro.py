# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

from data_manager import DataManager # such as DataManager

from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import VarianceThreshold


class Preprocessor(BaseEstimator):

    def __init__(self):

        self.transformer = VarianceThreshold(threshold=(.8 * (1 - .8)))


    def fit(self, X, y=None):

        return self.transformer.fit(X, y)


    def fit_transform(self, X, y=None):

        return self.transformer.fit_transform(X,y)


    def transform(self, X, y=None):

        return self.transformer.transform(X)
    
