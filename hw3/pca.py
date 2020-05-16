import numpy as np
data = np.loadtxt('/Users/pz/Desktop/inf552/hw3pca+fastmap/pca-data.txt')
def newX(data):
    colmean = np.mean(data,axis = 0)
    newX = data - colmean
    return newX,colmean
new_X,col_mean = newX(data)
n = 3
k = 2
cov = np.dot(new_X.T,new_X)/len(data)
eigenvalue,eigenvector = np.linalg.eig(cov)
sortVal = np.argsort(eigenvalue)
top_ = eigenvector[:,sortVal[:-k-1:-1]]
data_ = np.dot(new_X,top_)
print(data_)
