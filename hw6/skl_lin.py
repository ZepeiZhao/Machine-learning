import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

data = np.loadtxt("linsep.txt", dtype = "float", delimiter = ",")
X = data[:,0:2]
Y = data[:,-1]


clf = SVC(kernel="linear",C = 4)
clf.fit(X,Y)

print("weight:",clf.coef_[0])
print("intercept:",clf.intercept_)
print(clf.support_vectors_)

fig, ax = plt.subplots(figsize=(8,5))
ax.scatter(X[:,0],X[:,1],s=50, c="g", marker="o", label="support_v")
x1 = -(clf.coef_[0][0]*(-1)+clf.intercept_)/clf.coef_[0][1]
x2 = -(clf.coef_[0][0]*(1)+clf.intercept_)/clf.coef_[0][1]
ax.set_xlim(left=-0.5, right=1.5)
ax.set_ylim(bottom=-0.5, top=1.5)
ax.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],marker = "*",facecolors='red',s=100, edgecolors='k')
plt.plot([-1,1], [x1,x2])
plt.show()
