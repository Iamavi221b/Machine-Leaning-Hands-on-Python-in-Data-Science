# Simple Linear Regression

## Importing the libraries
## Importing the Libraries
import time 
start_time = time.time()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

## Spliting the datset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

## Training the Simple Linear Regression Model on the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

## Predicting the Test set results
y_pred = regressor.predict(X_test)

## Visualising the Training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary VS Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

## Visualising the Test set results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary VS Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

## Making a single prediction
print(regressor.predict([[12]]))
# Above will print salary of the individual with provided number of years of experience
###### Important Note ######
# Notice that the value of the feature (12 years) was a input in double pair of square brackets.
# That's because the 'predict' method always expects a 2D array as the format of its inputs.
# And puting 12 into a double pair of square brackets makes the input exactly a 2D array.
# Simply put:
# 12      ----- scaler
# [12]    ----- 1D array
# [[12]]  ----- 2D array

## Showing the coefficient , interception and score of the prediction
print(regressor.score(X_train, y_train))
print(regressor.coef_)
print(regressor.intercept_)

# Therefore, the equation of our simple linear regaression model is:
#  Salary = regressor.coef_ * Years of Experience + regressor.intercept_
###### Important Note ######
# To get those coefficients we called the 'coef_' and 'intercept_' attributes from our regressor
# object. Attributes in python are different than methods and usually return a simple value or
# an array of values.

print("----- %s seconds -----"% (time.time()-start_time))