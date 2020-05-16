#Zepei Zhao (5635405542)
#Mingjun Liu (1321657359)

import numpy as np
import cv2
from scipy.special import expit

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
        train_data = np.array(train_data,dtype = 'float')/255
        train_label = np.array(train_label)
    return train_data, train_label

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
        test_data = np.array(test_data,dtype = 'float')/255
        n = len(test_data)
    return test_data,test_actual

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

def train(x,y,epochs):
    for i in range(epochs):
        back(X,Y,w1,w2)
    return w1,w2

def forward(x):
    output1 = sigmoid(np.dot(x,w1)) #hidden layer
    output2 = sigmoid(np.dot(output1,w2)) #output layer
    return output1,output2

def back(X,Y,w1,w2):
    output1,output2= forward(X)
    o_w2 = np.dot(output1.T,((Y-output2)*sigmoid_derivative(output2)))
    dot = np.dot((Y - output2) * sigmoid_derivative(output2), w2.T)
    dot1 = dot*sigmoid_derivative(output1)
    h_w1 = np.dot(X.T,dot)
    w1 += np.multiply(learning_rate, h_w1)
    w2 += np.multiply(learning_rate, o_w2)
    return w1,w2

def predict(x,w1,w2): #(X)
    y = np.zeros((x.shape[0],w2.shape[1]))
    a = np.zeros((x.shape[0],w2.shape[1]))
    a,y = forward(x)
    return y

if __name__ == '__main__':
    input_layer_size = 960
    learning_rate = 0.1
    hidden_layer_size = 100
    output_layer_size = 1
    epochs = 1000
    w1 = np.random.uniform(-0.01, 0.01, size=(input_layer_size, hidden_layer_size))
    w2 = np.random.uniform(-0.01, 0.01, size=(hidden_layer_size, output_layer_size))
    X,Y = load_train_file()
    Y = np.reshape(Y, (184, 1))
    test_data, test_actual = load_test_file()
    w1_out, w2_out = train(X, Y, epochs)
    p = []
    prediction = predict(test_data, w1_out, w2_out)
    # print(prediction.shape)
    for i in range(len(prediction)):
        if prediction[i] <= 0.5:
            p.append(0)
        else:
            p.append(1)
    print(p)
    correct = 0
    for i in range(len(p)):
        if p[i] == test_actual[i]:
            correct += 1
    print("accuracy: ", correct / len(p))
    print("#correct items:", correct)
    #print(test_actual)








