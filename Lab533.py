import numpy as np
from sklearn import metrics, linear_model
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

data = pd.read_csv('Auto.csv', usecols=range(0,8), parse_dates=True)

df = pd.DataFrame({'mpg':data['mpg'],'horsepower':  pd.to_numeric(data['horsepower'])})

X_train = df[['horsepower']]
Y_train = df['mpg']

#Running cross_val_score for polynomial with different degrees
for i in range(1, 11):
    poly = PolynomialFeatures(degree=i)
    X_train_poly = poly.fit_transform(X_train)
    lm = linear_model.LinearRegression()
    #Using 5 folds
    k_fold = KFold(n_splits=10) 
    test = cross_val_score(lm, X_train_poly, Y_train, cv=k_fold, scoring = 'neg_mean_squared_error')
    print np.mean(-test)