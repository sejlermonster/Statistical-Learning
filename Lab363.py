import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, metrics

boston = datasets.load_boston()

lm = linear_model.LinearRegression()
boston_X_train = boston.data
boston_X_test = boston.data
#for i in range(0,13):
      #boston_X_train.append([item for sublist in boston.data[:, np.newaxis, i] for item in sublist]) 
      #boston_X_test.append([item for sublist in boston.data[:, np.newaxis, i] for item in sublist]) 

boston_Y_train = boston.target
boston_Y_test = boston.target

lm.fit(boston_X_train, boston_Y_train)

# The coefficients
print('Coefficients:')
print lm.coef_
print('Intercept: %.2f'
      % lm.intercept_)
# The mean squared error
# print("Mean squared error: %.2f"
#       % np.mean((lm.predict(boston_X_test) - boston_Y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Multiple R-Squared: %.2f' % lm.score(boston_X_test, boston_Y_test))



#plt.scatter(boston_x, boston_Y_test,  color='black')
#plt.plot(boston_X_test, lm.predict(boston_X_test), color='blue',
 #       linewidth=3)

# plt.xticks(())
# plt.yticks(())

# plt.show()