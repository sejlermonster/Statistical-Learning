#Linear discriinant analysis
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import pandas as pd

data = pd.read_csv('Smarket.csv', usecols=range(1,10), index_col=0, parse_dates=True)

X_train = data[:'2004'][['Lag1', 'Lag2']]
Y_train = data[:'2004']['Direction']

X_test = data['2005':][['Lag1', 'Lag2']]
Y_test = data['2005':]['Direction']

#knn = KNeighborsClassifier(n_neighbors=1)
knn = KNeighborsClassifier(n_neighbors=3)
fit = knn.fit(X_train,Y_train)

print "\n"
print"Confusion matrix:"
print(pd.crosstab(Y_test, knn.predict(X_test),rownames=['True'], colnames=['Predicted'], margins=True).T)

print "\n"
print "Score of train and test data"
print(knn.score(X_train, Y_train), knn.score(X_test, Y_test))

print "\n"
print "classification report"
print metrics.classification_report(Y_test, knn.predict(X_test))

# h = 0.02
# # Plot the decision boundary. For that, we will asign a color to each
# # point in the mesh [x_min, m_max]x[y_min, y_max].
# x_min, x_max = X_test['Lag1'].min() - .5, X_test['Lag1'].max() + .5
# y_min, y_max = X_test['Lag2'].min() - .5, X_test['Lag2'].max() + .5
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
# pl.figure(1, figsize=(4, 3))
# pl.set_cmap(pl.cm.Paired)
# pl.pcolormesh(xx, yy, Z)

# # Plot also the training points
# pl.scatter(X_test['Lag1'], X_test['Lag2'],c=Y )
# pl.xlabel('Sepal length')
# pl.ylabel('Sepal width')

# pl.xlim(xx.min(), xx.max())
# pl.ylim(yy.min(), yy.max())
# pl.xticks(())
# pl.yticks(())

# pl.show()







