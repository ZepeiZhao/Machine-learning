#Zepei Zhao (5635405542)
#Mingjun Liu (1321657359)

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def predict(data, w):
    prediction = np.zeros(len(data))
    for i in range(len(data)):
        #print('pre',w.dot(data[i]))
        if w.dot(data[i]) >= 0:
            prediction[i] = 1
        else:
            prediction[i] = -1
    return prediction

data = np.loadtxt('classification.txt',delimiter = ',')

points = data[:,0:3]
flag = data[:,-2]

alpha = 0.01
N, d = points.shape

weights = np.random.rand(3)
weights = np.append([0.01], weights)
#weights = np.zeros(4)

x0 = np.ones((N,1))
points = np.append(x0, points, axis = 1)

prediction = np.empty(N)
prediction[:] = -1

iteration = 0


while True:
    check = False
    iteration = iteration + 1
    for i in range(len(points)):
        wx = weights.dot(points[i])
        
        if flag[i] == 1 and wx < 0:
            weights = weights + alpha * points[i]
            #iteration += 1
            check = True
            break
        elif flag[i] == -1 and wx >= 0:
            weights = weights - alpha * points[i]
            #iteration += 1
            check = True
            break
    if iteration >= 7000:
        break
    if check ==False:
        break

testpoints = data[:,0:3]
x1 = np.ones((2000,1))
testpoints = np.append(x0, testpoints, axis = 1)
testflag = data[:,-2]

prediction = predict(testpoints, weights)
fail = 0
for i in range(len(testflag)):
    if testflag[i] == prediction[i]:
        continue
    else:
        fail = fail + 1  

print('weight:')
print(weights)
print('accuracy:')
print((len(testpoints)-fail)/len(testpoints))


#%%
"""
from sklearn.linear_model import Perceptron
X = data[:,0:3]
Y = data[:,-2]
clf = Perceptron(alpha = 0.01, max_iter = 7000)
clf.fit(X,Y)
print(clf.coef_)
print(clf.score(X,Y))
"""

