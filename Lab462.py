
#Logistic regression is a linear model for classification rather than regression
#This program is a binary classifier

#import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, metrics
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

data = pd.read_csv('Smarket.csv', usecols=range(1,10), index_col=0, parse_dates=True)
print data.head()

# Can be used to get summary including p-value
#formula = 'Direction ~ Lag1+Lag2+Lag3+Lag4+Lag5+Volume'
#model = smf.glm(formula=formula, data=data, family=sm.families.Binomial())
#result = model.fit()
#print(result.summary())

#X_train = data[:'2004'][['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']]
X_train = data[:'2004'][['Lag1', 'Lag2']]
Y_train = data[:'2004']['Direction']

#X_test = data['2005':][['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']]
X_test = data['2005':][['Lag1', 'Lag2']]
Y_test = data['2005':]['Direction']

lr = linear_model.LogisticRegression()
lr.fit(X_train,Y_train)

print "Score of train and test data"
print(lr.score(X_train, Y_train), lr.score(X_test, Y_test))

print "\n"
print"Confusion matrix:"
print(pd.crosstab(Y_test, lr.predict(X_test),rownames=['True'], colnames=['Predicted'], margins=True))

print "\n"
print "classification report"
print(metrics.classification_report(Y_test, lr.predict(X_test)))





