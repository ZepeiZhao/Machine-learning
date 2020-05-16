#Zepei Zhao (5*2)
#Mingjun Liu (1*9)

import numpy as np
import cvxopt
import matplotlib.pyplot as plt

#load data
data = np.loadtxt("linsep.txt", dtype = "float", delimiter = ",")
X = data[:,0:2]
Y = data[:,-1]
Y = Y.reshape(100,1)
m,n = X.shape

#qpp
Q = np.zeros((m,m))
for i in range(m):
    for j in range(m):
        Q[i,j] = np.dot(X[i],X[j])
P = cvxopt.matrix(np.outer(Y,Y)*Q)
q = cvxopt.matrix(np.ones(m)*(-1))
G = cvxopt.matrix(np.diag(np.ones(m)*(-1)))
h = cvxopt.matrix(np.zeros(m))
b = cvxopt.matrix(float(0))
A = cvxopt.matrix(Y.T)

sol = cvxopt.solvers.qp(P,q,G,h,A,b)
alpha = np.array(sol['x']).reshape(1,m)[0] #alpha:(N,1)

# sv
a = []
svx = []
svy = []
for i in range(len(alpha)):
    if alpha[i] > 0.00000001:
        a.append(alpha[i])
        svx.append(X[i])
        svy.append(Y[i])
a_ = np.array(a)
sv_x = np.array(svx)
sv_y = np.array(svy)
print("support_vectors: ", sv_x)

# w,b
w = np.zeros(n)
for i in range(len(a)):
    w += a_[i] * sv_x[i] * sv_y[i]
b = sv_y[0] - np.dot(w,sv_x[0])
print("w",w)
print("b",b)

# plot
fig, ax = plt.subplots(figsize=(8,5))
sx = []
sy = []
class1_x = []
class1_y = []
class2_x = []
class2_y = []
for i in range(m):
    if alpha[i]>0.00000001:
        sx.append(X[i,0])
        sy.append(X[i,1])
for i in range(m):
    if Y[i] == 1:
        class1_x.append(X[i,0])
        class1_y.append(X[i,1])
    else:
        class2_x.append(X[i,0])
        class2_y.append(X[i,1])

ax.scatter(sx, sy, s=150, c="g", marker="*", label="support_v")
ax.scatter(class1_x, class1_y, s=25, c="b", marker="o", label="class 1")
ax.scatter(class2_x, class2_y, s=30, c="r", marker="x", label="class -1")

ax.set_xlim(left=-0.5, right=1.5)
ax.set_ylim(bottom=-0.5, top=1.5)

l = -(w[0]*(-1)+b)/w[1]
r = -(w[0]*1+b)/w[1]
plt.plot([-1,1],[l,r])

plt.show()
