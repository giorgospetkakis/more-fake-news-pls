import os

import keras
import numpy as np
import pandas as pd
import sklearn.svm
from keras.layers import LSTM, Dense
from keras.models import Sequential
from matplotlib import pyplot as plt
from scipy.stats import mode
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    AdaBoostClassifier, AdaBoostRegressor, ExtraTreesClassifier,
    GradientBoostingClassifier, RandomForestClassifier)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, mean_squared_error)
from sklearn.model_selection import KFold, train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import BernoulliRBM, MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

import data
import xgboost as xgb
from utils import go_to_project_root


def classify(classifier, xtrain, ytrain, xtest, ytest):
    classifier.fit(xtrain, ytrain)
    imp = classifier.feature_importances_
    pred = predict(classifier, xtest)
    return pred

# def feature_elim(classifier, xtrain, ytrain):
#     rfe = RFE(estimator=classifier, n_features_to_select=n_features, step=10)
#     rfe.fit(xtrain, ytrain)
#     return rfe.ranking_

def get_new_xtests(path): 
    path = data_root[:-1] + "s/" + path
    xtest = []
    for file in os.listdir(path + "X_test/"):
        xtest += [pd.read_csv(path + "X_test/" + file, index_col=0).to_numpy()[:, [2, 3, 4, 5]]]
    return xtest

def read_data(_path):
    path = data_root + _path
    xtrain = pd.read_csv(path + "X_train.csv", index_col=0).to_numpy()[:200].astype(float)
    ytrain = pd.read_csv(path + "y_train.csv", index_col=0).to_numpy()[:200]
    ytest = pd.read_csv(path + "y_test.csv", index_col=0).to_numpy()
    xtest = []
    for file in os.listdir(path + "X_test/"):
        xtest += [pd.read_csv(path + "X_test/" + file, index_col=0).to_numpy()]

    new_xt = get_new_xtests(_path)
    mean_xt = [np.mean(xt[0], axis=0) for xt in new_xt]

    for i, x in enumerate(xtest):
        xtest[i][:,[2, 3, 4, 5]] = mean_xt[i]

    return xtrain, ytrain.ravel(), xtest, ytest.ravel()

def predict(classifier, xtest):
    majority_vote_preds = []
    for x in xtest:
        majority_vote_preds += [np.sum(classifier.predict(x).astype(int)) > 1]
    return majority_vote_preds

# Hardcode all the different feature sets
feature_sets = {
    "lexical": [0, 1],
    "semantic": [2, 3, 4, 5],
    "clusters": [6, 7],
    "nonling":  list(range(8, 30)),
    "pos": [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
    "emotion": [47, 48, 49, 50, 51, 52, 53, 54, 55, 56],
    "embeddings": list(range(57, 357))
}

go_to_project_root()
data_root = "data/processed/800/"
datasets = [read_data(f"K{k+1}/") for k in range(3)]

mean = 0
for i in range(3):
    xtrain, ytrain, xtest, ytest = datasets[i]
    c = GradientBoostingClassifier()
    pred = classify(c, xtrain, ytrain, xtest, ytest)
    acc = balanced_accuracy_score(ytest, pred)
    mean += acc
print(f"Accuracy: {mean / 3}")
