#Zepei Zhao
#USCID:5635405542
#Mingjun Liu
#USCID:1321657359

from matplotlib import pyplot as plt
from copy import deepcopy
import numpy as np
import pandas as pd


data = pd.read_csv('clusters.txt',header = None)
#print(data)
data = pd.DataFrame(data)
data = data.rename(index = str,columns = {0:'A',1:'B'})
#print(data.shape)
data.head()
A = data['A'].values
B = data['B'].values
X = np.array(list(zip(A,B)))

k = 3

#Assign centroids randomly
c1 = np.random.randint(0,np.max(X)-2,size = k)
c2 = np.random.randint(0,np.max(X)-2,size = k)
c = np.array(list(zip(c1,c2)),dtype = np.float32)
#print(c)

#Calculate distance with euclidean distance
def distance(x,y,ax = 1):
    return np.linalg.norm(x-y,axis=ax)

#Draw initial scatter and centroids
plt.rcParams['figure.figsize'] = (10,8) 
plt.style.use('seaborn')
plt.scatter(A,B,c = 'black',s=7)
plt.scatter(c1,c2,marker='*',s = 200,c = 'g')

c_old = np.zeros(c.shape)
clusters = np.zeros(len(X))
#print(c_old)

diff = distance(c,c_old,None)
while diff != 0:
    i = 0
    for i in range(len(X)):
        distances = distance(X[i],c) #Calculate the distance of each point and each cluster
        cluster = np.argmin(distances)
        clusters[i]=cluster #Assign each point to the cloest centroid
    c_old = deepcopy(c)
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j]==i]
        c[i] = np.mean(points,axis = 0) #Recompute centroid for each cluster
    diff = distance(c,c_old,None)
    i = i + 1

#print(type(clusters))

colors = ['r','g','b']
plt.subplots()
for i in range(k):
    points = np.array([X[j] for j in range(len(X))if clusters[j]==i])
    plt.scatter(points[:,0],points[:,1],s=20,c=colors[i])
plt.scatter(c[:,0],c[:,1],marker='*',s=200,c='black')

print('Centroids: ',c)
print('Iteration times: ',i)
plt.show()


