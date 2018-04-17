
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.cross_validation as sklearn


# Importing the dataset
dataset = pd.read_csv('mushrooms.csv')
X = dataset.iloc[:, 1: 23].values #we dont want the 'class' column as this is the dependant variable
y = dataset.iloc[:, 0].values #this is the dependant variable

"""# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)"""


#Encoding independant variable
"""from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray"""
X = pd.DataFrame(X)
X_dummies = pd.get_dummies(X)
y_dummies = pd.get_dummies(y)

# Avoiding dummy variable trap 
X_dummies = X_dummies.iloc[:, 1:].values
y_dummies = y_dummies.iloc[:, 1:].values


#split data for training and testing
from sklearn.cross_validation import  train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_dummies, y_dummies, test_size = 0.2, random_state = 0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Backward Elimination
import statsmodels.formula.api as sm
X_train = np.append(arr = np.ones((np.shape(X_train)[0],1)).astype(int), values = X_train, axis = 1)

