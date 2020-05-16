import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

data = np.loadtxt("nonlinsep.txt", dtype = "float", delimiter = ",")
X = data[:,0:2]
Y = data[:,-1]


clf = SVC(kernel="poly", degree = 2)
clf.fit(X,Y)
#
print("weights:",clf.dual_coef_[0])
print("intercept:",clf.intercept_)
print("support vectors:",clf.support_vectors_)


fig, ax = plt.subplots(figsize=(8,5))
ax.scatter(X[:,0],X[:,1],s=50, c="g", marker="o", label="support_v")
ax.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],marker = "*",facecolors='red',s=100, edgecolors='k')
plt.show()