import numpy as np
from sklearn import metrics, linear_model
from sklearn.model_selection import KFold, cross_val_score
import pandas as pd
import sklearn.linear_model as lm
from sklearn.feature_selection import RFE, SelectKBest
from sklearn.model_selection import train_test_split
from itertools import combinations
import matplotlib.pyplot as plt
import pylab
from sklearn import metrics
import math
import imp

metricsUtil = imp.load_source('MetricsUtil', 'MetricsUtil.py')

#Validation set approach 
data = pd.read_csv('Hitters.csv', usecols=range(0,21), parse_dates=True).dropna()
train, test = train_test_split(data, test_size = 0.5)

X_train = pd.get_dummies(train.drop('Salary', axis=1).drop('Name', axis=1)).drop(["League_A", "Division_E", "NewLeague_A"], axis=1)
Y_train = train['Salary']

X_test = pd.get_dummies(test.drop('Salary', axis=1).drop('Name', axis=1)).drop(["League_A", "Division_E", "NewLeague_A"], axis=1)
Y_test = test['Salary']

def process_subset(feature_set):
    d = len(feature_set)
    n = X_train.shape[1]
    # Fit model on feature_set  
    model = lm.LinearRegression().fit(X_train[[i for i in feature_set]], Y_train)
    Y_hat = model.predict(X_test[list(feature_set)])
    
    rss = metricsUtil.RSS(Y_test, Y_hat)  

    #mse = metrics.mean_squared_error(Y_test, Y_hat)
    mse = metricsUtil.Mse(Y, Y_hat)
    
    return {"model":model, 
            "RSS":rss,
            "features": feature_set,
            "mse": mse }

def best_subset_selection(k):
    results = []
    for combo in combinations(X_train.columns, k):
        results.append(process_subset(combo))
    models = pd.DataFrame(results)
    #Choose best model based on RSS
    best_model = models.loc[models['RSS'].argmin()]
    return best_model

subsets = pd.DataFrame(columns=["RSS", "model", "features", "mse"])

for i in xrange(1, 4):
    subsets.loc[i] = best_subset_selection(i)
    
RSS = subsets.RSS
features = subsets.features
mse = subsets.mse
print "features"
print features
print "mse"
print mse

X = pd.get_dummies(data.drop('Salary', axis=1).drop('Name', axis=1))
Y = data["Salary"]

kf = KFold(n_splits=10)
kf.get_n_splits(X)
#for i in range(1:19)