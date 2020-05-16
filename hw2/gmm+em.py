#Zepei Zhao
#USCID:5635405542
#Mingjun Liu
#USCID:1321657359

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import multivariate_normal

#Load data
data = np.loadtxt('clusters.txt',delimiter = ',')
#print(data[2])
data1 = np.split(data, [1], axis=1)
A = []
B = []
for each in data1[0]:
    A.append(each[0])
for each in data1[1]:
    B.append(each[0])

plt.rcParams['figure.figsize'] = (10,8)
plt.style.use('seaborn')
plt.scatter(A,B,c = 'black',s = 7)

#Initialize
n = len(data)
k = 3
W = np.ones((n,k))/k
Mu = np.random.rand(3,2)
Var = [[1, 1], [1, 1], [1, 1]]
Pi = [1/k]*k

#Gaussian distribution
def gaussian(data_small,mu,var): # 1*2, 1*2, [2*2]
    det=np.linalg.det(var)
    inv=np.linalg.inv(var)
    diff = np.matrix(data_small-mu)
    d = mu.shape[0]
    exp=math.exp((-0.5)*diff*inv*diff.T)
    g = ((2*math.pi)**(-0.5*d))*(det**(-0.5))*exp
    return g

#E-step update W
def expectation(data,Mu,Var,Pi): #150*2, 3*2, [2*2] * 3, 3*1, 150*3
    W = np.zeros((len(data),k))
    
    for j in range(len(data)):
        den = 0
        save_num = []
        for i in range(k):
            var = np.zeros((2,2))
            np.fill_diagonal(var,Var[i])
            #print(Var[i])
            num = Pi[i]*gaussian(data[j,:],Mu[i],var)
            den = den + num
            save_num.append(num)
        
        W[j,:] = save_num/den   
        
    return W

#M-step update Mu,Var,Pi
def maximization(W,data):
    Mu = np.zeros((k,2))
    Var = np.ones((k,2))
    for i in range(k):
        Pi = W.sum(axis = 0)/W.sum()
        Mu[i] = np.average(data,axis = 0,weights = W[:,i])
        Var[i] = np.average((data-Mu[i])**2,axis = 0,weights = W[:,i])
    return Pi,Mu,Var

#Iterate until converge to threshold
iteration = 1
threshold = 1e-6
while True:
    old_W = W
    W = expectation(data,Mu,Var,Pi)
    Pi,Mu,Var = maximization(W,data)
    print('Iteration times: ',iteration,'Mu: ',Mu,'Var:',Var,'Pi: ',Pi)
    
    iteration += 1
    diff = old_W -W
    if diff.var() < threshold:
        break
    if iteration > 1000:
        break
#plot
C = W.argmax(axis = 1)
cluster1 = np.array([data[i] for i in range(n) if C[i] == 0])
cluster2 = np.array([data[i] for i in range(n) if C[i] == 1])
cluster3 = np.array([data[i] for i in range(n) if C[i] == 2])
plt.scatter(cluster1[:,0],cluster1[:,1],s = 7,c = 'Red')
plt.scatter(cluster2[:,0],cluster2[:,1],s = 7,c = 'Green')
plt.scatter(cluster3[:,0],cluster3[:,1],s = 7,c = 'Blue')
plt.show()
