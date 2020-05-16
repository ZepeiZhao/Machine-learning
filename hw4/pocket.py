#Zepei Zhao (5635405542)
#Mingjun Liu (1321657359)

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import time 

start = time.time()

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
flag = data[:,-1]

alpha = 0.01
N, d = points.shape

weights = np.random.rand(3)
weights = np.append([0.01], weights)
#weights = np.zeros(4)

x0 = np.ones((N,1))
points = np.append(x0, points, axis = 1)

prediction = np.empty(N)
prediction[:] = -1

bestpass = 7000
bestweight = np.zeros(4)

iteration = 0
results = []

while True:
    check = False #boolean to determine meet all constraints
    update = True #boolean to determine update weight
    violated = 0
    
    if iteration >= 7000:
        break
    for i in range(len(points)):

        wx = weights.dot(points[i])
        
        if flag[i] == 1 and wx < 0:
            violated = violated + 1
            if update == True:
                weights = weights + alpha * points[i]
                check = True
                update = False
       
        elif flag[i] == -1 and wx >= 0:
            violated = violated + 1
            if update == True:  
                weights = weights - alpha * points[i]
                check = True
                update = False
        
    results.append(violated)
    iteration = iteration + 1
    
    if violated < bestpass:
        bestpass = violated
        bestweight = weights      
        
    if check == False:
        break

testpoints = data[:,0:3]
x1 = np.ones((2000,1))
testpoints = np.append(x0, testpoints, axis = 1)
testflag = data[:,-1]

prediction = predict(testpoints, bestweight)
fail = 0
for i in range(len(testflag)):
    if testflag[i] == prediction[i]:
        continue
    else:
        fail = fail + 1 
        

print('best weight:')
print(bestweight)
print('accuracy:')
print((len(testpoints)-fail)/len(testpoints))

plt.style.use('seaborn')
plt.figure()
plt.xlabel('Iterations')
plt.ylabel('Number of Misclassifications')
plt.plot(list(range(1,7001)), results)
plt.show()



#%%
"""
from sklearn.linear_model import Perceptron
X = data[:,0:3]
Y = data[:,-2]
clf = Perceptron()
clf.fit(X,Y)
clf.coef_
"""

