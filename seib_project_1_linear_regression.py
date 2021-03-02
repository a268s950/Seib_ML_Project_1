import numpy as np
import pandas as pd
import csv
from math import sqrt
import matplotlib.pyplot as plt
import itertools

df2 = pd.read_csv('Salary.csv')  #read_csv vs DataFrame gets it in the form we want to pull out X and Y columns
df2.head()

X = df2.iloc[:, :-1].values
Y = df2.iloc[:, -1].values

def mean(values):
    return sum(values)/float(len(values))

def variance(values,mean):
    return sum([(X-mean)**2 for x in values])

def covariance(X, mean_x, Y, mean_y):
    covar = 0.0
    for i in range(len(X)):
        covar += (X[i] - mean_x)*(Y[i]-mean_y)
    return covar

def coefficients(df2):
    B1 = covariance(X, mean_x, Y, mean_y) / variance(X, mean_x)
    B0 = mean_y - B1* mean_x
    return [B0, B1]

def linear_regression(train, test):
    predictions = list()
    yhat = B0_f + B1_f * X #X is years experience
    predictions.append(yhat)
    return predictions

def rmse_measured(actual, predicted):
    sum_error = 0.0
    for i in range(len(predicted)): #was actual but apparently one list is longer than the other lo
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = (sum(sum_error)) / float(len(actual))
    return sqrt(mean_error)

mean_x, mean_y = mean(X), mean(Y)
var_x, var_y = variance(X, mean_x), variance(Y, mean_y)

mean_x_value, var_x_value = mean_x[0], var_x[0]
#mean_y_value = mean_y[0] #it says "IndexError: invalid index to scalar variable" so I just took out
var_y_value = var_y[0]

covar = covariance(X, mean_x, Y, mean_y)

print('Covariance: %.3f' % (covar))
print('x stats: mean=%.3f variance=%.3f' % (mean_x_value, var_x_value))
print('y stats: mean=%.3f variance=%.3f' % (mean_y, var_y_value))
B0, B1 = coefficients(df2) #type type 'numpy.ndarray'
B0_f, B1_f = B0[0][0], B1[0][0] #type 'numpy.float64'
print('Coefficients: B0=%.3f, B1=%.3f' % (B0_f, B1_f))
test_set = list()
rmse = linear_regression(df2, test_set)
rmse_me = rmse_measured(Y,rmse)
resize_rmse = [elem for twod in rmse for elem in twod]

plt.scatter(X, Y, color='red')
plt.plot(X, resize_rmse)
plt.show()
