import numpy as np
from sklearn import metrics, linear_model
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import statsmodels.api as sm
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from itertools import combinations

# drop rows with nan values
data = pd.read_csv('Hitters.csv', usecols=range(0,21), parse_dates=True).dropna()
print list(data.columns.values)

X = pd.get_dummies(data.drop('Salary', axis=1).drop('Name', axis=1))
Y = data['Salary']

# lm = linear_model.LinearRegression()
# rfe = RFE(lm, 8)
# rfe = rfe.fit(estimtor=np.mean((regr.predict(X) - Y) ** 2), X, Y)
# ((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2).sum())
# selector = SelectKBest(score_func=sklearn.feature_selection.f_regression, k=8)
# selector.fit(X, Y)
selector = SelectKBest(f_classif, k=19)
selector.fit(X, Y)
scores = selector.scores_
X_train = selector.fit_transform(X, Y)
selector.get_support()
print scores

# print(rfe.support_)
# print(rfe.ranking_)


