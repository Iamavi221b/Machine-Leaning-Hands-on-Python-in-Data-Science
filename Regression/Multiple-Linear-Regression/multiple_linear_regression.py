# Multiple Linaer Regression

## Importing the libraries
import time
start_time = time.time()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Importing the Dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

## Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

## Splitting dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

## Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

## Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

## Making a single prediction
## e.g : Startup with R&D Spend = 160000, Administration Spend = 130000, 
## Marketing Spend = 300000 and State = California
print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]]))
## IMPORTANT NOTE 1: Notice that the values of the features were all input in a double pair of
## square brackets. That's because the "predict" method expects a 2D array as the format of its
## inputs. And putting our values into a double pair of square brackets makes the input exactly
## a 2D array. Simply put:
## 1, 0, 0, 160000, 130000, 300000      --->    scalars
## [1, 0, 0, 160000, 130000, 300000]    --->    1D array
## [[1, 0, 0, 160000, 130000, 300000]]  --->    2D array

## IMPORTANT NOTE 2: Notice also that the "California" state was not input as a string in the 
## last column but as 1, 0, 0 in the first 3 columns. That's beacuse of course the predict method
## expects the one-hot-encoded value of the state, and as we see  in the second row of the matrix
## of features X, California was encoded as 1, 0 , 0. And be careful to include these values in
## the first three columns, not the last three ones, because the dummy variables are always
## created in the first columns.

## Getting Final Linear Regression Equation with the values of the cofficients
print(regressor.coef_)
print(regressor.intercept_)

## Therefore, the equation of our multiple linear regression model is:
## Profit = value_1 * Dummy state 1 + value_2 * Dummy state 2 + value_3 * Dummy state 3 +
## value_4 * R&D Spend + value_5 * Administration + value_6 * Marketing Spend + Intercept

## IMPORTANT NOTE: To get these coefficients we called the "coef_" and "intercept_" attributes
## from our regressor object. Attributes in python are different then methods and usually return
## a simple value or an array of values.

print("------ %s seconds ------"%(time.time() - start_time))