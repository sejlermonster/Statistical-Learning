import numpy as np
import pandas as pd
from random import randint
from random import randint

def boot_python(data, function, num_of_iteration):
    n = data.shape[0]
    idx = np.random.randint(0, n, (num_of_iteration, n))
    stat = np.zeros(num_of_iteration)
    for i in xrange(len(idx)):
        stat[i] = function(data, idx[i])
    
    return {'Mean': np.mean(stat), 'std. error': np.std(stat)}

def alpha_fn(data, index):
    X = data['X'][index]
    Y = data['Y'][index]
    return ((np.var(Y) - np.cov(X,Y)) / (np.var(X)+np.var(Y) - 2*np.cov(X,Y)))[0,1]

def boot_fn(data, index):
    X = data['horsepower'][index]
    Y = data['mpg'][index]
    return ((np.var(Y) - np.cov(X,Y)) / (np.var(X)+np.var(Y) - 2*np.cov(X,Y)))[0,1]


portData = pd.read_csv('Portfolio.csv', usecols=range(1,3), parse_dates=True)

print alpha_fn(portData, range(0,100))
print alpha_fn(portData, np.random.choice(range(0, 100), size=100, replace=True))
print boot_python(portData, alpha_fn, 1000)

autoData = pd.read_csv('Auto.csv', usecols=range(0,8), parse_dates=True)
print boot_fn(autoData, np.random.choice(range(0, 392), size=100, replace=True))

