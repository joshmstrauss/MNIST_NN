# MNIST Digit Recognizer Neural Network

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print("Current working directory:", os.getcwd())
BASE_DIR = os.path.dirname(__file__)
file_path = os.path.join(BASE_DIR, "train.csv")
data = pd.read_csv('train.csv.zip', compression='infer')

# prepping data
data = np.array(data)
m, n = data.shape #where m is number of rows and n is number of columns
np.random.shuffle(data)

data_dev = data[0:1000].T #.T to transpose
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

data_train = data [1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]

# Normalize the data
X_train = X_train / 255.0
X_dev = X_dev / 255.0

def init_params():
    W1 = np.random.randn(10,784) * np.sqrt(1. / 784)
    b1 = np.zeros((10,1))
    W2 = np.random.randn(10,10) * np.sqrt(1. / 10)
    b2 = np.zeros((10,1))
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    Z_stable = Z - np.max(Z, axis=0, keepdims=True)
    return np.exp(Z_stable) / np.sum(np.exp(Z_stable), axis=0, keepdims=True)

def forward_pass(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_ReLU(Z):
    return Z > 0

def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    Y = Y.flatten()
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1/ m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1/ m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha): #where alpha is the learning rate
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_pass(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            print("Accuracy: ", get_accuracy(get_predictions(A2), Y))
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 200, .1)