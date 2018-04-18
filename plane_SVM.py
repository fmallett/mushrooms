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
cl = svm.SVC(C = 0.1 )
cl.fit(X_train, y_train)

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


"""    
#Copy paste
le=preprocessing.LabelEncoder()
for col in data.columns:
    data[col]=le.fit_transform(data[col])
    
X1=data.iloc[:,1:23]
y1=data.iloc[:,0]

scaler = preprocessing.StandardScaler()
X1=scaler.fit_transform(X1)

from sklearn import svm
from sklearn.model_selection import cross_val_score
clf_linear2 = svm.LinearSVC( C=0.5)
scores = cross_val_score(clf_linear2, X1, y1, cv=5)
scores          

from sklearn import svm
from sklearn.model_selection import cross_val_score
clf_linear2 = svm.LinearSVC( C=1)
scores = cross_val_score(clf_linear2, X1, y1, cv=5)
scores   

#SVC with linear kernel 
clf_l=svm.SVC(kernel='linear')
clf_l.fit(X1,y1)
scores = cross_val_score(clf_l, X1, y1, cv=5)
scores

clf_l.coef_

clf_nl=svm.SVC()
clf_nl.fit(X1,y1)
scores = cross_val_score(clf_nl, X1, y1, cv=5)
scores

from sklearn.metrics import accuracy_score

import numpy as np
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=.33, random_state=42)

clf_nl=svm.SVC(C=1)
clf_nl.fit(X_train,y_train)
y_pred = clf_nl.predict(X_test)
scores=accuracy_score(y_test,y_pred)
scores

clf_nl=svm.SVC(C=10,gamma=10,kernel='poly')


clf_nl.fit(X_train,y_train)
y_pred = clf_nl.predict(X_test)
scores=accuracy_score(y_test,y_pred)
scores
"""