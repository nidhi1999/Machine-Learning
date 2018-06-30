# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, 1].values

#missing data

#encoding categorical data

#splitting data into test set and training set
from sklearn.model_selection import train_test_split
X_test,X_train,Y_test,Y_train=train_test_split(X, Y, test_size=1/3, random_state=0)
#feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)"""
#fitting simple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
#predicting the linear model
Y_pred=regressor.predict(X_test)

plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience(training set)')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()
#visualising test set results

plt.scatter(X_test, Y_test, color='red')
plt.plot(X_train,regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience(training set)')
plt.xlabel('years of experience')
plt.ylabel('salary') 
plt.show()