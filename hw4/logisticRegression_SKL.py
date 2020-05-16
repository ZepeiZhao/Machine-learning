#Zepei Zhao (5635405542)
#Mingjun Liu (1321657359)

import numpy as np
from sklearn.linear_model import LogisticRegression

data = np.loadtxt('classification.txt',delimiter=",",usecols=(0,1,2,4))
X = data[:,[0,1,2]]
Y = data[:,3]
N = len(X[0])
model = LogisticRegression(max_iter=7000,fit_intercept = True)
model.fit(X,Y)

prediction = model.predict(X)
correct = np.where(Y==prediction)[0].shape[0]
accuracy = correct/prediction.shape[0]

print("accuracy = ",accuracy)
print(model.coef_)
print(model.intercept_)

