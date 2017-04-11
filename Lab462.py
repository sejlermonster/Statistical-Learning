#Logistic regression is a linear model for classification rather than regression
#This program is a binary classifier

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from scipy import stats


f = open("Smarket.csv")
f.readline()  # skip the header
data = np.genfromtxt(f, usecols=(1, 2, 3, 4, 5, 6, 7, 8), delimiter=',')



# We look at the probability for "Up" and "Down".
# Using logistic regression we get a probability between [0,1] 