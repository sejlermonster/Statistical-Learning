import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from sklearn import metrics, linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

data = pd.read_csv('Auto.csv', usecols=range(0,8), parse_dates=True)

df = pd.DataFrame({'mpg':data['mpg'],'horsepower':  pd.to_numeric(data['horsepower'])})
train, test = train_test_split(df, test_size = 0.5)

X_train = train[['horsepower']]
Y_train = train['mpg']

X_test = test[['horsepower']]
Y_test = test['mpg']

#Linear regression
lm = linear_model.LinearRegression()
lm.fit(X_train, Y_train)

print("Mean squared error: %.2f"
      % np.mean((lm.predict(X_test) - Y_test) ** 2))

# Polynomial regression with 2 degrees
poly = PolynomialFeatures(degree=2)
X_train_poly2 = poly.fit_transform(X_train)
X_test_poly2 = poly.fit_transform(X_test)

lm = linear_model.LinearRegression()
lm.fit(X_train_poly2, Y_train)

print("Mean squared error: %.2f"
      % np.mean((lm.predict(X_test_poly2) - Y_test) ** 2))

# Polynomial regression with 3 degrees
poly = PolynomialFeatures(degree=3)
X_train_poly3 = poly.fit_transform(X_train)
X_test_poly3 = poly.fit_transform(X_test)

lm = linear_model.LinearRegression()
lm.fit(X_train_poly3, Y_train)

print("Mean squared error: %.2f"
      % np.mean((lm.predict(X_test_poly3) - Y_test) ** 2))



