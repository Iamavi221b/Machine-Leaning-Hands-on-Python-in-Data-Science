# Data Preprocessing Tools

## Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
start_time = time.time()

## Importing the dataset

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(X, y)

print("----- %s seconds -----"% (time.time()-start_time))