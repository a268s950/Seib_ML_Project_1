import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import math

df2 = pd.read_csv('avocado.csv')
X = df2.iloc[:, :-1].values
Y = df2.iloc[:, -12].values

def cdf(Y):
    x1, counts = np.unique(Y, return_counts=True)
    cusum = np.cumsum(counts)
    print(cusum[-1])
    return (x1, (cusum.astype(float))/cusum[-1]) 

def normal_distribution(Y , mean , std):
    prob_density = (np.pi*std) * np.exp(-0.5*((Y-mean)/std)**2)
    return prob_density

#mean = 0
mean = sum(Y)/len(Y)
var = sum(pow(x-mean,2) for x in Y)/len(Y)
std = math.sqrt(var)
print(mean)

x2, y2 = cdf(Y)
x2 = np.insert(x2,0, x2[0])

y2 = np.insert(y2, 0, 0.)
print(y2)
plt.plot(x2, y2, drawstyle='steps-post')
plt.grid(True)
plt.savefig('cdf.png')

#Plotting the Results
#plt.scatter(Y, pdf, color = 'red') #WHY DID THIS SUDDENLY BREAK AND HOW DID I FORGET WHAT SHOULD BE THERE INSTEAD OF PDF!!!!!
#plt.xlabel('Data points')
#plt.ylabel('Probability Density')
#plt.show()
