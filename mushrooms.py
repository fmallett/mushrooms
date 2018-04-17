import pandas as pd 
import numpy as np 

dataset = pd.read_csv('/Users/matthiasboeker/Desktop/mushrooms.csv')

def splitdata(dataset , alpha):
    dataset = dataset
    alpha = alpha
    
    size = np.shape(dataset)
    trainsize = int(np.round(size[0]*0.8))
    X_train = dataset.iloc[0:trainsize,1:size[1]].values
    y_train = dataset.iloc[:,0].values
    
    X_test = dataset.iloc[trainsize+1:size[0], 1:size[1]].values
    y_test = dataset.iloc[trainsize+1:size[0], 0].values
    
    return X_train, y_train, X_test, y_test


[X_train, y_train, X_test, y_test] = splitdata(dataset, 0.8)

"Create Dummy-Variables"

X_train = pd.DataFrame(dataset)
dummies = pd.get_dummies(dataset)

"""from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X = labelencoder_X.fit_transform(X_train[:,0:np.shape(X_train)[1]])
onehotencoder = OneHotEncoder(categorical_features = "all")
X = onehotencoder.fit_transform(X)"""
