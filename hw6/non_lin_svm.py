#Zepei Zhao (5*2)
#Mingjun Liu (1*9)

import numpy as np
import cvxopt
import matplotlib.pyplot as plt
import math

data = np.loadtxt("nonlinsep.txt", dtype = "float", delimiter = ",")
X = data[:,0:2]
Y = data[:,-1]
Y = Y.reshape(100,1)
m,n = X.shape

fig = plt.figure()
ax = plt.subplot()
class1_x = []
class1_y = []
class2_x = []
class2_y = []
for i in range(m):
    if Y[i] == 1:
        class1_x.append(X[i,0])
        class1_y.append(X[i,1])
    else:
        class2_x.append(X[i,0])
        class2_y.append(X[i,1])
ax.scatter(class1_x, class1_y, s=25, c="b", marker="o", label="class 1")
ax.scatter(class2_x, class2_y, s=30, c="r", marker="x", label="class -1")
#plt.show()

def kernal(x,y):
    #z = [1,x**2,y**2]
    z = [1, x ** 2, y ** 2, math.sqrt(2) * x, math.sqrt(2) * y, math.sqrt(2) * x * y]
    return z
print("kernel function: k(x,x’) = (1+xTx’)2")
Z = np.zeros((m,6))
for i in range(m):
    Z[i] = kernal(X[i][0],X[i][1])
a,b = Z.shape

Q = np.zeros((a,a))
for i in range(a):
    for j in range(a):
        Q[i,j] = np.dot(Z[i],Z[j])
P = cvxopt.matrix(np.outer(Y,Y)*Q)
q = cvxopt.matrix(np.ones(a)*(-1))
G = cvxopt.matrix(np.diag(np.ones(a)*(-1)))
h = cvxopt.matrix(np.zeros(a))
b = cvxopt.matrix(float(0))
A = cvxopt.matrix(Y.T)

sol = cvxopt.solvers.qp(P,q,G,h,A,b)
alpha = np.array(sol['x']).reshape(1,a)[0]

al = []
svx = []
svy = []
for i in range(len(alpha)):
    if alpha[i] > 0.000000001:
        al.append(alpha[i])
        svx.append(X[i])
        svy.append(Y[i])
a_ = np.array(al)
sv_x = np.array(svx)
sv_y = np.array(svy)
print("support_vectors: ", sv_x)

w = np.zeros(n)
for i in range(len(al)):
    w += a_[i] * sv_x[i] * sv_y[i]
b_ = sv_y[0] - np.dot(w,sv_x[0])
print("w",w)
print("b",b_)
sx = []
sy = []
for i in range(m):
    if alpha[i]>0.0000000001:
        sx.append(X[i,0])
        sy.append(X[i,1])
ax.scatter(sx, sy, s=100, c="y", marker="*", label="support_v",alpha = 0.6)
plt.show()


