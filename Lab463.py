#Linear discriinant analysis
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics
import pandas as pd

data = pd.read_csv('Smarket.csv', usecols=range(1,10), index_col=0, parse_dates=True)

X_train = data[:'2004'][['Lag1', 'Lag2']]
Y_train = data[:'2004']['Direction']

X_test = data['2005':][['Lag1', 'Lag2']]
Y_test = data['2005':]['Direction']

lda = LinearDiscriminantAnalysis()
fit = lda.fit(X_train,Y_train)

print "Prior probabilities:"
print lda.priors_

#Not the same as in the book
print "\n"
print "Coefficients:"
print lda.coef_

print "\n"
print "Mean:"
print lda.means_

print "\n"
print"Confusion matrix:"
print(pd.crosstab(Y_test, lda.predict(X_test),rownames=['True'], colnames=['Predicted'], margins=True))

print "\n"
print "Score of train and test data"
print(lda.score(X_train, Y_train), lda.score(X_test, Y_test))

print "\n"
print "classification report"
print metrics.classification_report(Y_test, lda.predict(X_test))







