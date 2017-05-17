import numpy as np
from sklearn import metrics, linear_model
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import statsmodels.api as sm
import sklearn.linear_model as lm
from sklearn.feature_selection import RFE, SelectKBest
from itertools import combinations
import matplotlib.pyplot as plt
import pylab
from sklearn import metrics
import math

# drop rows with nan values
data = pd.read_csv('Hitters.csv', usecols=range(0,21), parse_dates=True).dropna()
print list(data.columns.values)

X = pd.get_dummies(data.drop('Salary', axis=1).drop('Name', axis=1))
Y = data['Salary']

#Residual sum of squares
def RSS(Y, Y_hat):
    residual = Y- Y_hat
    return (residual ** 2).sum()

# def Cp(RSS, len_of_featureset, Y_hat, X, Y):
#     return ( RSS + 2 * len_of_featureset * Y_hat.var() ) / X.shape[1]
          
# def Bic(X, RSS, len_of_featureset):
#     n = X.shape[1]
#     return n * math.log(RSS / n) + len_of_featureset * math.log(n)

def process_subset(feature_set):
    d = len(feature_set)
    # Fit model on feature_set  
    regr = lm.LinearRegression().fit(X[[i for i in feature_set]], Y)
    Y_hat = regr.predict(X[list(feature_set)])
    
    rss = RSS(Y, Y_hat)   
    rsquared = regr.rsquared
    
    # cp = Cp(rss, d, Y_hat, X, Y)
    # bic = Bic(X, rss, d)
    model = sm.OLS(Y,X[list(feature_set)])
    regr = model.fit()
    cp = regr.aic # same as cp when doing linear regression
    bic = regr.bic
    
    rsquared = metrics.r2_score(Y, Y_hat)
    return {"model":regr, 
            "RSS":rss,
            "features": feature_set,
            "rsquared": rsquared,
            "bic": bic,
            "cp":cp }

def get_best(k):
    results = []
    for combo in combinations(X.columns, k):
        results.append(process_subset(combo))
    models = pd.DataFrame(results)
    best_model = models.loc[models['RSS'].argmin()]
    return best_model

subsets = pd.DataFrame(columns=["RSS", "model", "features", "rsquared", "bic", "cp"])
for i in xrange(1,3):
    subsets.loc[i] = get_best(i)

rsquared = subsets.rsquared
RSS = subsets.RSS
bic = subsets.bic
cp = subsets.cp
features = subsets.features
print "features"
print features
print "RSS"
print RSS
print "Rsquared"
print rsquared
print "cp"
print cp
print "bic"
print bic


plt.subplot(2, 2, 1)
plt.plot(RSS)
plt.xlabel('# Predictors')
plt.ylabel('RSS')

plt.subplot(2, 2, 2)
plt.plot(rsquared)
plt.xlabel('# Predictors')
plt.ylabel('adjusted rsquared')

plt.subplot(2, 2, 3)
plt.plot(cp)
plt.xlabel('# Predictors')
plt.ylabel('CP')

plt.subplot(2, 2, 4)
plt.plot(bic)
plt.xlabel('# Predictors')
plt.ylabel('BIC')

plt.show()