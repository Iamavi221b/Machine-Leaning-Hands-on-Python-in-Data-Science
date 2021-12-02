# Ploynomial Regression

## Importing the libraries
import time
from matplotlib import colors
start_time = time.time()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Importing the datset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

## Training the linear regression model on the whole dataset
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)

## Training the polynomial regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
polynomial_regressor = PolynomialFeatures(degree=4)
X_polynomial = polynomial_regressor.fit_transform(X)
linear_regressor_2 = LinearRegression()
linear_regressor_2.fit(X_polynomial, y)

## Visualising the linear regression results
plt.scatter(X, y, color='red')
plt.plot(X, linear_regressor.predict(X), color='blue')
plt.title('Truth or Bluff (Linear Regresion)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

## Visualising the polynomial regression results
plt.scatter(X, y, color='red')
plt.plot(X, linear_regressor_2.predict(X_polynomial), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

## Visualising the polynomial regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')
plt.plot(X, linear_regressor_2.predict(X_polynomial), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

## Predicting a new result with Linear Regression
print(linear_regressor.predict([[6.5]]))

## Predicting a new result with Polynomial Regression
print(linear_regressor_2.predict(polynomial_regressor.fit_transform([[6.5]])))

print("----- %s seconds -----"%(time.time() - start_time))