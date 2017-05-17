import math
import pylab
import numpy as np
import pandas as pd
import statsmodels.api as sm
import sklearn.linear_model as lm
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn import metrics, linear_model



# drop rows with nan values
data = pd.read_csv('Hitters.csv', usecols=range(0,21), parse_dates=True).dropna()
print list(data.columns.values)

X = pd.get_dummies(data.drop('Salary', axis=1).drop('Name', axis=1))
Y = data['Salary']

#Residual sum of squares
def RSS(Y, Y_hat):
    residual = Y - Y_hat
    return (residual ** 2).sum()

#Residual sum of squares
def A_rsquare(RSS, Y, Y_hat, n, d):
    #Total sum of squares, formula: sum((y-y_mean)^2)
    TSS = ((Y - Y.mean()) ** 2).sum()
    #Formula from slides:  adjusted R^2 = 1-(RSS/(n-d-1)/(TSS/(n-1)))
    return 1.0 - ((RSS / (n - d - 1)) / (TSS / (n - 1)))

def Cp(RSS, d, Y_hat, n, Y):
    # Formula from slides: Cp = 1/n * (RSS + 2* d * sigma^2)
    return  (1.0 / n) * (RSS + 2 * d * Y_hat.var())
              
def Bic(n, RSS, d, Y_hat):
      # Formula from slides: bic = 1/n * (RSS+log(n)*d*sigma^2)
      return (1.0 / n) * (RSS + math.log(n) * d * Y_hat.var())

def process_subset(feature_set):
    d = len(feature_set)
    n = X.shape[1]
    # Fit model on feature_set  
    regr = lm.LinearRegression().fit(X[[i for i in feature_set]], Y)
    Y_hat = regr.predict(X[list(feature_set)])
    
    rss = RSS(Y, Y_hat)  
    rsquared = metrics.r2_score(Y, Y_hat)
    #rsquared = A_rsquare(rss, Y, Y_hat, n, d)

    cp = Cp(rss, d, Y_hat, n, Y)
    bic = Bic(n, rss, d, Y_hat)
    
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

def backward(predictors):
    results = []
    for combo in combinations(predictors, len(predictors)-1):
        results.append(process_subset(combo))
    models = pd.DataFrame(results)
    best_model = models.loc[models['RSS'].argmin()]
    return best_model

def forward(predictors):
    remaining_predictors = [p for p in X.columns if p not in predictors]
    results = []
    for p in remaining_predictors:
        results.append(process_subset(predictors+[p]))
    models = pd.DataFrame(results)
    best_model = models.loc[models['RSS'].argmin()]
    return best_model

subsets = pd.DataFrame(columns=["RSS", "model", "features", "rsquared", "bic", "cp"])

#Best subset selection
# for i in xrange(1,3):
#     subsets.loc[i] = get_best(i)

# #Forward stepwise selection
# predictors = []
# for i in xrange(1, len(X.columns)+1):
#     subsets.loc[i] = forward(predictors)
#     predictors = subsets.loc[i].features
    
#Backward stepwise selection
predictors = X.columns
while(len(predictors) > 1):
    subsets.loc[len(predictors)-1] = backward(predictors)
    predictors = subsets.loc[len(predictors)-1].features

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