#Zepei Zhao (5635405542)
#Mingjun Liu (1321657359)

import numpy as np
from sklearn.linear_model import LinearRegression

data = np.loadtxt('linear-regression.txt',delimiter=",",usecols=(0,1,2))
X = data[:,[0,1]]
a = np.ones((1))
X = np.insert(X,0,values = a,axis = 1)
Y = data[:,2]

model = LinearRegression()
model.fit(X,Y)
print(model.coef_)
print(model.intercept_)
