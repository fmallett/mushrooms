#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 12:35:17 2018

@author: matthiasboeker
"""
import pandas as pd 

data = pd.read_csv('mushrooms.csv')

X = data.iloc[:,1:23]
y = data.iloc[:,0]
XX = data.iloc[:,1:23]

import numpy as np
from sklearn.model_selection import train_test_split

#X = X.values
#y = y.values

#Either dummy or scaling 

#Introducing dummy variables

X = pd.get_dummies(X)
y = pd.get_dummies(y)

#Avoid dummy trap 
X = X.iloc[:,1:].values
y = y.iloc[:,1:].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33, random_state = 1)

#Fitting to SVC model
from sklearn import svm
cl = svm.SVC(C = 0.1)
cl.fit(X_train, y_train)

#Optimizing the parameters of the SVM
#Spezifiy parameters and distributions to sample
parameters = {'kernel':('linear', 'rbf'),
                'C':[0.0001, 0.001, 0.01, 0.1, 1],
                'gamma':[0.0001, 0.001, 0.01, 0.1, 1]
              }
#Randomized Search Method
from sklearn.model_selection import RandomizedSearchCV

#Run randomized search
n_iter_search = 5
random_search = RandomizedSearchCV(cl, param_distributions=parameters, n_iter=n_iter_search)
random_search.fit(X,y)

#Grid Search Method
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(cl, parameters)
grid_search.fit(X, y) #iterate over all configurations


#Fitting test data
y_predict = cl.predict(X_test)


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
score = accuracy_score(y_test, y_predict)
cm = confusion_matrix(y_test, y_predict)
print(score)
print(cm)

#Fitting to linear model 
cl_linear = svm.LinearSVC()
cl_linear.fit(X_train,y_train)


y_predict_linear = cl_linear.predict(X_test)
score = accuracy_score(y_test, y_predict_linear)
print(score)




#we’re fitting the model on the training data and trying to predict the test data
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)

#plot the model
## The line / model
plt.scatter(y_test, predictions)
plt.xlabel('True values')
plt.ylabel('Predictions')

#accuracy score
print('Score:')
print(model.score(X_test, y_test))


from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics

#Cross Validation
scores = cross_val_score(model, XX, y, cv=3)
print('Cross-validated scores:', scores)

predictions = cross_val_predict(model, XX, y, cv=3)
plt.scatter(y, predictions)
