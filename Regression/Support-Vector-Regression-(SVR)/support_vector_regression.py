# Support Vector Regression (SVR)

## Importing the libraries
import time
from matplotlib import colors
start_time = time.time()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Importing Dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:,-1].values

y = y.reshape(len(y),1)
## reshaping the value to 2-D array since feature scaling takes 2-D array as input value.

## Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
## Here we have created different scale for X and y since they both have different range and
## distribution of value which are very different from one another.
X = sc_x.fit_transform(X)
y = sc_y.fit_transform(y)


## Training SVR model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
## rbf stands for radial basis function kernel, here while using SVR class we need to decide
## the kernel which we are going to use.
regressor.fit(X, y)


## Predicting new result
print(sc_y.inverse_transform([regressor.predict(sc_x.transform([[6.5]]))]))
## since we have apply feature scaling to all the variable we need to transform the observe
## value into same scale.

## Visualising the SVR results
plt.scatter(sc_x.inverse_transform(X), sc_y.inverse_transform(y), color='red')
## here we need to inverse the scaled value to real value in order to crate a graph.
y_pred = regressor.predict(X)
y_pred = y_pred.reshape(len(y_pred), 1)
plt.plot(sc_x.inverse_transform(X), sc_y.inverse_transform(y_pred) , color='blue')
## here we need to inverse the scaled value to real value in order to plot the graph
plt.title('Truth or Bluff - Support Vector Regression SVR')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

## Visualising the SVR results (for higher resolution and smother curve) 
X_grid = np.arange(min(sc_x.inverse_transform(X)), max(sc_x.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(sc_x.inverse_transform(X), sc_y.inverse_transform(y), color='red')
## here we need to inverse the scaled value to real value in order to crate a graph.
y_pred = regressor.predict(X_grid)
y_pred = y_pred.reshape(len(y_pred), 1)
print(sc_y.inverse_transform(y_pred))
plt.plot(X_grid, sc_y.inverse_transform(y_pred) , color='blue')
## here we need to inverse the scaled value to real value in order to plot the graph
plt.title('Truth or Bluff - Support Vector Regression SVR')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

print("----- %s seconds -----"%(time.time() - start_time))