#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 10:19:29 2018

@author: matthiasboeker
"""

#Data preprocessing function

    #Only appliable for mushroom data
    #Transformes categorical data by standardization or transformation to dummy variables 
    #Input: data, switch: (Choose between standardization (0) or dummy variables (1)), test_size, random_state
    #Output:  X_train, X_test, y_train, y_test

import pandas as pd 
import numpy as np
 
def data_preprocessing(data, switch = 0 ,test_size = 0.33, random_state = None):
    from sklearn.model_selection import train_test_split
    
    if switch == 0:  
       from sklearn import preprocessing
       label_encoder =preprocessing.LabelEncoder()
       for i in data.columns:
           data[i] = label_encoder.fit_transform(data[i])
                   
       scaler = preprocessing.StandardScaler()
       data =scaler.fit_transform(data)
       X = data[:,1:23]
       y = data[:,0]

       X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.5, random_state = random_state )
       return(X_train, X_test, y_train, y_test)
            
    elif switch == 1:
         X = data.iloc[:,1:23]
         y = data.iloc[:,0]
         X = pd.get_dummies(X)
         y = pd.get_dummies(y)
         
         #Avoid dummy trap 
         X = X.iloc[:,1:].values
         y = y.iloc[:,1:].values
         X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.5, random_state = random_state)
         return(X_train, X_test, y_train, y_test)
    else:
        print('ERROR: Input of switch variable must be 0 or 1')
    