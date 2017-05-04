import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from sklearn import metrics, linear_model
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv('Auto.csv', usecols=range(0,8), parse_dates=True)

df = pd.DataFrame({'mpg':data['mpg'],'horsepower':  pd.to_numeric(data['horsepower'])})
train, test = train_test_split(df, test_size = 0.5)

X_train = train[['horsepower']]
Y_train = train['mpg']

X_test = test['horsepower']
Y_test = test['mpg']

lm = linear_model.LinearRegression()
lm.fit(X_train, Y_train)
print("Mean squared error: %.2f"
      % np.mean((lm.predict(X_test) - Y_test) ** 2))

print('Variance score: %.2f' % lm.score(X_train, Y_train))

print "something"