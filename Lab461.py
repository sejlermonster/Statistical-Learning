#Logistic regression is a linear model for classification rather than regression
#This program is a binary classifier

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from scipy import stats


f = open("Smarket.csv")
f.readline()  # skip the header
data = np.genfromtxt(f, usecols=(1, 2, 3, 4, 5, 6, 7, 8), delimiter=',')

# prepare date for correlation function
preparedData = []
for i in range(0,8):
    preparedData.append([item[i] for item in data])


# Look for correlation between predictors
cor = np.corrcoef(preparedData)
print "Year             Lag1          Lag2          Lag3          Lag4       Lag5       Volume      Today"
print cor