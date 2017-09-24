"""
Cameron Knight
Intro and data section od the Machien learning series by Sentdex
https://www.youtube.com/channel/UCfzlCWGWYyIQ0aLC5w48gBQ
Description: Basic introduction in data and linear regression learning on stoke data from quandl
"""
import pandas as pd
import quandl
import math
import datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

df = quandl.get('WIKI/GOOGL') #Gets Google stock price from online database

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']] #Sets the colums of data that are relevant
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/ df['Adj. Close'] * 100.0 # Gets the %offset of the high to the close value
df['PCT_change'] = (df['Adj. Open'] - df['Adj. Close'])/ df['Adj. Close'] * 100.0# Gets the % offset of the Open to Close value

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']] #Sets the new relevant columns

forcast_col = 'Adj. Close' # Sets the column used to forcast can be changed in other data sets allows for VVV that code to be re-used

df.fillna(-99999, inplace = True)# Replaces al NAN data with data cuz data is needed at every point or error

forcast_out = int(math.ceil(0.02 * len(df))) # Days to shift data to predict


df['lable'] = df[forcast_col].shift(-forcast_out)


X = np.array(df.drop(['lable'],1))# Features gives everything excpt lables
X  = preprocessing.scale(X) # Normalizes x across the rest of the data
X_lately = X[-forcast_out:] # Gets the set of current dated ofset by the forcast shift
X = X[:-forcast_out]


df.dropna(inplace = True) # Drops lines of NA
Y = np.array(df['lable']) # Lables gives only lables

# Shuffles and splits the data int training and testing data storing them in the variables thusly
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X,Y,test_size = 0.2)  

#n_jobs is the nuber of threads to break up the process into LinearReg can be wholy threaded
clf = LinearRegression(n_jobs = -1) #sets the function to linear regression 

clf.fit(X_train,Y_train) # runs the data in the linear regresion model





# Read and write using pickle Cannot Be used
# with open('linnearregression.pickle','wb') as f:
# 	pickle.dump(clf,f)

# pickle_in = open('linnearregression.pickle','wb')
# clf = pickle.load(pickle_in)

#End Read Write



accuracy = clf.score(X_test,Y_test) #compares the traind network against new data recording to accuracy

print("Offset:", forcast_out)
print("Accuracy:", accuracy)


forcast_set = clf.predict(X_lately) # oredidics the stock prices for X_Lately

print("Forcast", forcast_set)


df['Forcast'] = np.nan


### PLOTS THE FORCAST
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forcast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day
	df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

df['Adj. Close'].plot()
df['Forcast'].plot()
plt.legend(loc = 4)
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()