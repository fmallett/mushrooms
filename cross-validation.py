# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 19:23:15 2018

@author: Fiona Mallett
Student Number: 23289339
"""


#Cross-validation to check how well the classifier performs7
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder



scaler = StandardScaler()
X_std = scaler.fit_transform(X)
le = LabelEncoder()
y_enc = le.fit_transform(y)


param_grid = [{'C': np.logspace(-3, 3, 10)}]



grid_search = GridSearchCV(
    estimator=LogisticRegression(),
    param_grid=param_grid,
    cv=StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42),
    scoring='f1',
    n_jobs=-1
)

scores = cross_val_score(
    estimator=grid_search,
    X=X_std,
    y=y,
    cv=StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=0),
    scoring='f1',
    n_jobs=-1
)
print('- - - CROSS VALIDATION - - -')