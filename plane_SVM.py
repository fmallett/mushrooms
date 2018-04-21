#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 12:35:17 2018

@author: matthiasboeker
"""
import pandas as pd 
import numpy as np

data = pd.read_csv('mushrooms.csv')

#Data preprocessing 
X_train, X_test, y_train, y_test = data_preprocessing(data,0, 0.33, 123)

#Fitting to SVC model
from sklearn import svm
cl = svm.SVC()

#Optimizing the parameters of the SVM
from sklearn.metrics import classification_report
#Spezifiy parameters and distributions to sample
parameters = {'kernel': ('linear', 'rbf'),
                'C':[ 0.01, 0.1, 1],
                'gamma':[0.0001, 0.001, 0.01, 0.1, 1]
              }
#Randomized Search Method
from sklearn.model_selection import RandomizedSearchCV

#Run randomized search
n_iter_search = 5
random_search = RandomizedSearchCV(cl, param_distributions=parameters, n_iter=n_iter_search)
random_search.fit(X,y)
print("Best parameters:", random_search.best_params_)
print(classification_report(y_test, random_search.predict(X_test))) 

#Grid Search Method
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(cl, parameters)
grid_search.fit(X, y) #iterate over all configurations
print("Best parameters:", grid_search.best_params_)
print(classification_report(y_test, grid_search.predict(X_test))) 

#Fitting to SVC model with optimized parameters
clf = svm.SVC(random_search.best_params_['C'],random_search.best_params_['kernel'],3,random_search.best_params_['gamma'])
clf.fit(X_train,y_train)


#Fitting test data
y_predict = clf.predict(X_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
score = accuracy_score(y_test, y_predict)
cm = confusion_matrix(y_test, y_predict)
print(score)
print(cm)


from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics


# Cross-Validation Curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.clf()
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

cv = ShuffleSplit(n_splits=10, test_size=0.33) #shuffle and split into n subsets
plot_learning_curve(cl, "Cross-validation on mushroom dataset", X, y, (0.7, 1.01), cv=cv, n_jobs=4)
plt.show()