#Zepei Zhao (5*2)
#Mingjun Liu (1*9)

import numpy as np

data = np.loadtxt('classification.txt',delimiter=",",dtype = "float",usecols=(0,1,2,4))
X = data[:,[0,1,2]]
a = np.ones((1,2000))
X = np.insert(X,0,values = a,axis = 1)
Y = data[:,3]
Y = Y[:,np.newaxis]
N = len(X)
w = np.random.rand(1,len(X[0]))
alpha = 0.1

# Iteration
i = 0
while i < 7000:
    s = np.multiply(np.dot(X,w.T),Y)
    sigmoid = 1/(1+np.exp(s))
    temp = np.sum(np.multiply(sigmoid,np.multiply(X,Y)),axis = 0)
    delta = -1*temp/N
    w -= alpha*delta
    i += 1

#Count accuracy
count = 0
prediction = np.ones(N)
f = np.dot(X,w.T)
for i in range(N):
    if f[i][0]<0.5:
        y = -1
    else:
        y = 1
    prediction[i] = y
    if prediction[i] != Y[i]:
        count += 1
accuracy = (N-count)/N

print("accuracy=",accuracy)
print(w)
