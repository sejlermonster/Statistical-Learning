import numpy as np
from sklearn import metrics, linear_model
from sklearn.model_selection import KFold, cross_val_score
import pandas as pd

data = pd.read_csv('Auto.csv', usecols=range(0,8), parse_dates=True)

df = pd.DataFrame({'mpg':data['mpg'],'horsepower':  pd.to_numeric(data['horsepower'])})

X_train = df[['horsepower']]
Y_train = df['mpg']

#Linear regression
lm = linear_model.LinearRegression()
lm.fit(X_train, Y_train)

print "\n"
print "Intercept:"
print lm.intercept_

print "\n"
print "Coefficients:"
print lm.coef_

#k_fold = KFold(n_splits=X_train.shape[0]) 
test = cross_val_score(lm, X_train, Y_train, cv=192,  scoring = 'neg_mean_squared_error', n_jobs=-1)
print np.mean(-test)