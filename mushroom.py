
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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray


labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray

"""#Encoding the dependant variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)"""







#split data for training and testing
from sklearn.cross_validation import  train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Fitting classifier to the Training set
#random_state parameter in SVC function is set to 0 to have the same results in group
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0, C = 1)
classifier.fit(X, y)



##-------------Wo
# Visualising the classifying resultss
from matplotlib.colors import ListedColormap
X_set, y_set = X, y
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier Two Spirals Task 1a (C = 1)')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.legend()
plt.show()