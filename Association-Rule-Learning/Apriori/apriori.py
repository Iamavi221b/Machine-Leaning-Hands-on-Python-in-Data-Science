# Apriori

## Importing the libraries
import time
start_time = time.time()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

## Data preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
transaction = []
for i in range(0, 7501):
    transaction.append([str(dataset.values[i,j]) for j in range(0, 20)])

## Training the apriori model on the dataset
from apyori import apriori
rules = apriori(transactions=transaction, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2,max_length=2)

## Visualising the results
## Displaying the first results coming directly from the output of the apriori function
results = list(rules)
print(results)

## Putting the results well organised into a Pandas DataFrame
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

## Displaying the result non sorted
print(resultsinDataFrame)

## Displaying the result sorted by descending lifts
print(resultsinDataFrame.nlargest(n=10, columns='Lift'))

print("----- {} seconds -----".format(time.time()-start_time))
