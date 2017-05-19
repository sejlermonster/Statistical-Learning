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

def process_subset(feature_set, x_train, y_train, x_test, y_test ):
    d = len(feature_set)
    n = y_test.shape[0]
    # Fit model on feature_set  
    model = lm.LinearRegression().fit(x_train[[i for i in feature_set]], y_train)
    Y_hat = model.predict(x_test[list(feature_set)])
    
    rss = metricsUtil.RSS(y_test, Y_hat)  

    #mse = metrics.mean_squared_error(y_test, Y_hat)
    mse = metricsUtil.Mse(n, y_test, Y_hat)
    
    return {"model":model, 
            "RSS":rss,
            "features": feature_set,
            "mse": mse }

def best_subset_selection(k, x_train, y_train, x_test, y_test ):
    results = []
    for combo in combinations(x_train.columns, k):
        results.append(process_subset(combo, x_train, y_train, x_test, y_test))
    models = pd.DataFrame(results)
    #Choose best model based on RSS
    best_model = models.loc[models['RSS'].argmin()]
    return best_model

subsets = pd.DataFrame(columns=["RSS", "model", "features", "mse"])

# ********* Validation split ************
# train, test = train_test_split(data, test_size = 0.5)

# X_train = pd.get_dummies(train.drop('Salary', axis=1).drop('Name', axis=1)).drop(["League_A", "Division_E", "NewLeague_A"], axis=1)
# Y_train = train['Salary']

# X_test = pd.get_dummies(test.drop('Salary', axis=1).drop('Name', axis=1)).drop(["League_A", "Division_E", "NewLeague_A"], axis=1)
# Y_test = test['Salary']

# for i in xrange(1, 4):
#     subsets.loc[i] = best_subset_selection(i, X_train, Y_train, X_test, Y_test)
    

# *****   Cross validation *******
X = pd.get_dummies(data.drop('Salary', axis=1).drop('Name', axis=1)).drop(["League_A", "Division_E", "NewLeague_A"], axis=1)
Y = data["Salary"]

kf = KFold(n_splits=10, shuffle=True)
kf.get_n_splits(X)
j = 0
mse = np.zeros((kf.n_splits, 19))
for train_index, test_index in kf.split(X):
    print "run"
    X_train = X.iloc[train_index]
    Y_train = Y.iloc[train_index]
    X_test = X.iloc[test_index]
    Y_test = Y.iloc[test_index]
    for i in xrange(1, 3):
         subset = best_subset_selection(i, X_train, Y_train, X_test, Y_test)
         mse[j,i-1] = subset["mse"]
    j = j+1
print mse.mean(axis=0)    


# ***** Print  features and mse *****
RSS = subsets.RSS
features = subsets.features
mse = subsets.mse
print "features"
print features
print "mse"
print mse
