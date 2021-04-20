# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 13:03:10 2018

@author: KD
"""

# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Importing dataset 
dataset = pd.read_csv('FullDataset.csv')
dataset.head()
# Removing call column
dataset1 = [col for col in dataset.columns if "call" not in col] # Removing call clumns and putting it in another variable
dataset = dataset[dataset1] # Transposing the rows into columns
dataset.head()
dataset.T.head()
dataset = dataset.T

dataset2 = dataset.drop(['Gene Description','Gene Accession Number'],axis=0)
dataset2.index = pd.to_numeric(dataset2.index)
dataset2.sort_index(inplace=True)
dataset2.head()

dataset2['cat'] = list(pd.read_csv('actual.csv')['cancer'])
dic = {'ALL':0,'AML':1}
dataset2.replace(dic,inplace=True)

X = dataset2.iloc[:, 0:7129].values
y = dataset2.iloc[:, 7129].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35, random_state = 0)

"""# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)"""

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 6)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_


# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier(base_score = 0.01, booster = 'gblinear',  gamma = 0.0001, n_estimators = 40)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_xg = confusion_matrix(y_test, y_pred)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', min_samples_leaf = 2, min_samples_split = 0.1)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Creating the matrix
from sklearn.metrics import confusion_matrix
cm_randomf = confusion_matrix(y_test, y_pred)

# Adaboost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import RandomForestClassifier

classifier = AdaBoostClassifier(RandomForestClassifier(max_depth=1, criterion = 'gini'),
                         algorithm="SAMME.R",
                         n_estimators=50)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test).astype(int)

# Creating the matrix
from sklearn.metrics import confusion_matrix
cm_ada = confusion_matrix(y_test, y_pred)



# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'algorithm': ['SAMME.R'], 'n_estimators': [3, 5, 10, 20, 30, 40, 50, 60, 70]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'recall',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_


# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 8))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the third hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))


# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 3, nb_epoch = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_ann = confusion_matrix(y_test, y_pred)