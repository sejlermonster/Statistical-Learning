#Linear discriinant analysis
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import metrics
import pandas as pd

data = pd.read_csv('Smarket.csv', usecols=range(1,10), index_col=0, parse_dates=True)

X_train = data[:'2004'][['Lag1', 'Lag2']]
Y_train = data[:'2004']['Direction']

X_test = data['2005':][['Lag1', 'Lag2']]
Y_test = data['2005':]['Direction']

qda = QuadraticDiscriminantAnalysis()
fit = qda.fit(X_train,Y_train)

print "Prior probabilities:"
print qda.priors_

print "\n"
print "Confusion matrix"
print metrics.confusion_matrix(Y_test, qda.predict(X_test))

print "\n"
print "Score of train and test data"
print(qda.score(X_train, Y_train), qda.score(X_test, Y_test))

print "\n"
print "classification report"
print metrics.classification_report(Y_test, qda.predict(X_test))







