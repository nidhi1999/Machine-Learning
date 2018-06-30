# Polynomial Regression

# Importing the dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
Y=dataset.iloc[:,2].values

"""from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2, random_state=0)"""
#fitting linear regression to dataset
from  sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,Y)
#fitting polynomial regression to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)
lin_reg2=LinearRegression()
lin_reg2.fit(X_poly, Y)
#visualising dataset in linear regression
plt.scatter(X, Y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title("truth or bluff(LR)")
plt.xlabel('level')
plt.ylabel('salaries')
plt.show()
#visualising dataset in polynomial regression
X_grid=np.arange(min(X), max(X), 0.1)
X_grid=X_grid.reshape((len(X_grid)), 1)
plt.scatter(X, Y, color='red')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title("truth or bluff(PR)")
plt.xlabel('level')
plt.ylabel('salaries')
plt.show()
#predicting salary(linear regression)
lin_reg.predict(6.5)
#predicying salary (polynomial regression)
lin_reg2.predict(poly_reg.fit_transform(6.5))