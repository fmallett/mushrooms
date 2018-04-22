#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 12:35:17 2018

@author: matthiasboeker
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('mushrooms.csv')

#Data preprocessing 
X_train, X_test, y_train, y_test = data_preprocessing(data,1, 0.99)

#Fitting to SVC model
from sklearn import svm
cl = svm.SVC()

#Optimizing the parameters of the SVM
#Spezifiy parameters and distributions to sample
parameters = {'kernel': ('linear', 'rbf'),
                'C':[ 0.001, 0.01, 0.1, 1],
                'gamma':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
              }
#Optimizing parameters 
best_para = para_opti(cl, X_train, y_train, X_test, y_test, parameters, 1 , 10)
    
#Fitting to SVC model with optimized parameters
clf = svm.SVC(best_para['C'],'rbf',3,best_para['gamma'], probability = True)
clf.fit(X_train,y_train)


#Fitting test data
y_predict = clf.predict(X_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
score = accuracy_score(y_test, y_predict)
cm = confusion_matrix(y_test, y_predict)
print(score)
print(cm)


# calculate the fpr and tpr for all thresholds of the classification
from sklearn import metrics
probs = clf.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

# plot ROC curve
plt.title('HWA1-C Receiver Operating Characteristic of SVM')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.4f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.plot([0, 0], [0, 1], c = 'grey')
plt.plot([0, 1], [1, 1], c = 'grey')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


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
plot_learning_curve(cl, "Cross-validation on mushroom dataset", X_test, y_test, (0.7, 1.01), cv=cv, n_jobs=4)
plt.show()