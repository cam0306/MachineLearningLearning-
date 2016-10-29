"""
Cameron Knight
Code based on k nearest neibors od the Machien learning series by Sentdex
https://www.youtube.com/channel/UCfzlCWGWYyIQ0aLC5w48gBQ
Description: my regression program
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, neighbors

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999,inplace = True)
del df['id']


X = np.array(df.drop(['class'],axis = 1))
y = np.array(df['class'])

X_trian, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size = 0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_trian,y_train)

accuracy = clf.score(X_test,y_test)
print(accuracy)

example_measures = np.array([[4,2,1,1,1,2,3,2,1],[12,4,1,9,1,2,3,2,1]])
example_measures = example_measures.reshape(len(example_measures),-1)
prediction = clf.predict(example_measures)

print(prediction)