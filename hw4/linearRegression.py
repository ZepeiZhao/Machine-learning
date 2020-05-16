#Zepei Zhao (5635405542)
#Mingjun Liu (1321657359)

import numpy as np

data = np.loadtxt('linear-regression.txt',delimiter=",",usecols=(0,1,2))
X = data[:,[0,1]]
a = np.ones((1))
X = np.insert(X,0,values = a,axis = 1)
Y = data[:,2]
w = np.random.rand(1,len(X[0]))

W1 = np.linalg.inv(np.dot(X.T,X))
W2 = np.dot(X.T,Y)
w = np.dot(W1,W2)
print(w)

