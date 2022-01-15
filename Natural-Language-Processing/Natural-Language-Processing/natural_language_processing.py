# Natural Language Processing

## Importing the librries
from ntpath import join
import random
import time
from traceback import print_tb
start_time = time.time()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

## Importing the dataset
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter='\t', quoting=3)

## Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

## Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

## Splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

## Training the Naive Bayes model on the training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

## Predicting the test set results
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

## Making the confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

## Predicting if a single review is postive or negative
my_review = 'I love this place'
my_review = re.sub('[^a-zA-Z]', ' ', my_review)
my_review = my_review.lower()
my_review = my_review.split()
my_review = [ps.stem(word) for word in my_review if not word in set(all_stopwords)]
my_review = ' '.join(my_review)
my_review = [my_review]
my_review = cv.transform(my_review).toarray()

print(classifier.predict(my_review))

print("----- {} seconds -----".format(time.time()-start_time))