#Zepei Zhao (5*2)
#Mingjun Liu (1*9)

from sklearn.linear_model import Perceptron
X = data[:,0:3]
Y = data[:,-2]
clf = Perceptron(alpha = 0.01, max_iter = 7000)
clf.fit(X,Y)
print(clf.coef_)
print(clf.score(X,Y))
