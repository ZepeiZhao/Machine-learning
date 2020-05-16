#Zepei Zhao (5*2)
#Mingjun Liu (1*9)

import numpy as np
import cv2
from sklearn.neural_network import MLPClassifier

def read_file(filepath):
    data = []
    for file in filepath:
        img = cv2.imread(file,-1)
        img_size = len(img)*len(img[0])
        reshape_img = list(img.reshape(img_size))
        data.append(reshape_img)
    return data

def load_train_file():
    filepath = []
    train_label = []
    with open('downgesture_train.list.txt') as f:
        for row in f.readlines():
            filepath.append(row.strip())
        for i in filepath:
            if 'down' in i:
                train_label.append(1)
            else:
                train_label.append(0)
        train_data = read_file(filepath)
        train_data = np.array(train_data)
        train_label = np.array(train_label)
    return train_data, train_label

X,Y = load_train_file()

#test_data: ((83, 960)
def load_test_file():
    filepath = []
    test_actual = []
    with open('downgesture_test.list.txt') as f:
        for row in f.readlines():
            filepath.append(row.strip())
        for i in filepath:
            if 'down' in i:
                test_actual.append(1)
            else:
                test_actual.append(0)
        test_data = read_file(filepath)
        test_data = np.array(test_data)

    return test_data,test_actual
test_data,test_actual = load_test_file()
n = len(test_data)
NN = MLPClassifier(hidden_layer_sizes=(100,),activation="logistic",solver="lbfgs",learning_rate_init=0.1,max_iter=1000)
NN.fit(X,Y)
correct = 0
prediction = NN.predict(test_data)
for i in range(n):
    if prediction[i] == test_actual[i]:
        correct += 1
print("accuracy: ",correct/n)
print("#correct items: ",correct)

