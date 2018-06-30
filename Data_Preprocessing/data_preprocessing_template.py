# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, 3].values

#missing data

#encoding categorical data

#splitting data into test set and training set
from sklearn.model_selection import train_test_split
X_test,X_train,Y_test,Y_train=train_test_split(X, Y, test_size=0.2, random_state=0)
#feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)"""
