# Decision Tree Regression

## Importing the libraries
import time
start_time = time.time()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

## Training the Decision Tree Regression model on the whole dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)
## Feature scaling is not required in decision tree regression since prediction from decision
## is a result of terminal leaf split not some equation so no feature scaling is required. 

## Predicting a new result
print(regressor.predict([[6.5]]))

## Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position Label')
plt.ylabel('Salary')
plt.show()

print("----- %s seconds -----"%(time.time() - start_time))