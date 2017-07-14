
#Logistic regression is a linear model for classification rather than regression
#This program is a binary classifier

#import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, metrics
import statsmodels.api as sm


#X_train = data[:'2004'][['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']]
#X_train = np.array([[1, 3], [2, 2], [3, -1], [3, -1], [5,2], [6,4], [10,1], [11,0]]).reshape(8,2)
X_train = np.array([[-2, -1, -5, 3, 2, 0]]).reshape(6,1)
#X_train = np.array([[1, 2, 3]]).reshape(3,1)
Y_train = np.array([0, 0, 0, 1, 1, 1])


# logit_mod = sm.Logit(Y_train, X_train)
# logit_res = logit_mod.fit()
# print logit_res.summary()

lr = linear_model.LogisticRegression()
lr.fit(X_train,Y_train,)

print lr.predict(2)
print lr.predict(-2)

print "\n"
print "Intercept:"
print lr.intercept_

print "\n"
print "Coefficients:"
print lr.coef_
