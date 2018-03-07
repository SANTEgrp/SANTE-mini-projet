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
from os.path import isfile
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

class model:
    def __init__(self):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation.
        '''
        self.num_train_samples=0
        self.num_feat=1
        self.num_labels=1
        self.is_trained=False
        self.mymodel=RandomForestClassifier(n_estimators = 500)

    def fit(self, X, y):
        '''
        This function should train the model parameters.
        Here we do nothing in this example...
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
            y: Training label matrix of dim num_train_samples * num_labels.
        Both inputs are numpy arrays.
        For classification, labels could be either numbers 0, 1, ... c-1 for c classe
        or one-hot encoded vector of zeros, with a 1 at the kth position for class k.
        The AutoML format support on-hot encoding, which also works for multi-labels problems.
        Use data_converter.convert_to_num() to convert to the category number format.
        For regression, labels are continuous values.
        '''
        self.num_train_samples = len(X)
        if X.ndim>1: self.num_feat = len(X[0])
        print("FIT: dim(X)= [{:d}, {:d}]").format(self.num_train_samples, self.num_feat)
        num_train_samples = len(y)
        if y.ndim>1: self.num_labels = len(y[0])
        print("FIT: dim(y)= [{:d}, {:d}]").format(num_train_samples, self.num_labels)
        if (self.num_train_samples != num_train_samples):
            print("ARRGH: number of samples in X and y do not match!")
        else:
            self.mymodel = self.mymodel.fit(X, y)
            self.is_trained=True

    def predict(self, X):
  
        return self.mymodel.predict_proba(X)[:,1]

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile) as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self

