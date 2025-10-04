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
    H1 = 256 # first hidden layer size (W2, b2)
    H2 = 128 # second hidden layer size (W3, b3)

    W1 = np.random.randn(H1,784) * np.sqrt(2.0 / 784)
    b1 = np.zeros((H1,1))

    W2 = np.random.randn(H2,H1) * np.sqrt(2.0 / H1)
    b2 = np.zeros((H2,1))

    W3 = np.random.randn(10,H2) * np.sqrt(2.0 / H2)
    b3 = np.zeros((10,1))

    return W1, b1, W2, b2, W3, b3

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    Z_stable = Z - np.max(Z, axis=0, keepdims=True)
    return np.exp(Z_stable) / np.sum(np.exp(Z_stable), axis=0, keepdims=True)

def forward_pass(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)

    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)

    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)

    return Z1, A1, Z2, A2, Z3, A3

num_classes = 10

def one_hot(Y):
    Y = Y.flatten().astype(int)
    one_hot_Y = np.zeros((num_classes, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y

def deriv_ReLU(Z):
    return (Z > 0).astype(Z.dtype)

def cross_entropy_loss(A3, Y):
    # A3: softmax outputs, shape (10, batch)
    # Y: integer labels, shape (batch,) or (1, batch)
    Y = Y.flatten().astype(int)
    # pick the predicted probability for the true class of each sample
    p_true = A3[Y, np.arange(A3.shape[1])]
    # add tiny epsilon to avoid log(0)
    eps = 1e-12
    return -np.mean(np.log(p_true + eps))

def back_prop(Z1, A1, Z2, A2, Z3, A3, W2, W3, X, Y):
    Y = Y.flatten()
    m = Y.size
    one_hot_Y = one_hot(Y)

    dZ3 = A3 - one_hot_Y
    dW3 = 1 / m * dZ3.dot(A2.T)
    db3 = 1 / m * np.sum(dZ3, axis=1, keepdims=True)

    dZ2 = W3.T.dot(dZ3) * deriv_ReLU(Z2)
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1/ m * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1/ m * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2, dW3, db3

def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha): #where alpha is the learning rate
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    W3 = W3 - alpha * dW3
    b3 = b3 - alpha * db3
    return W1, b1, W2, b2, W3, b3

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    # print(predictions, Y)    # prints predictions every few iterations, slows down output
    return np.sum(predictions == Y) / Y.size

def gradient_descent_minibatch(X, Y, X_dev, Y_dev, epochs=10, batch_size=128, alpha=0.01, log_every=100):
    W1, b1, W2, b2, W3, b3 = init_params()
    m = X.shape[1]  # number of training samples (columns)

    step = 0
    for epoch in range(1, epochs + 1):
        # 1) shuffle indices each epoch
        indices = np.random.permutation(m)

        # 2) loop over mini-batches
        for start in range(0, m, batch_size):
            end = min(start + batch_size, m)
            batch_idx = indices[start:end]

            Xb = X[:, batch_idx]        # (784, batch)
            Yb = Y[batch_idx]           # (batch,)

            # forward → backprop → update (for this batch)
            Z1, A1, Z2, A2, Z3, A3 = forward_pass(W1, b1, W2, b2, W3, b3, Xb)
            dW1, db1, dW2, db2, dW3, db3 = back_prop(Z1, A1, Z2, A2, Z3, A3, W2, W3, Xb, Yb)
            W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3,
                                                    dW1, db1, dW2, db2, dW3, db3, alpha)

            # progress log every N mini-batches
            if step % log_every == 0:
                loss = cross_entropy_loss(A3, Yb)
                acc = get_accuracy(get_predictions(A3), Yb)
                print(f"Step {step:5d} | Train loss {loss:.4f} | Train acc {acc:.4f}")
            step += 1

        # 3) end of epoch: evaluate on the dev set
        _, _, _, _, _, A3_dev = forward_pass(W1, b1, W2, b2, W3, b3, X_dev)
        dev_acc = get_accuracy(get_predictions(A3_dev), Y_dev)
        print(f"Epoch {epoch:2d} done | Dev acc {dev_acc:.4f}")

    return W1, b1, W2, b2, W3, b3


W1, b1, W2, b2, W3, b3 = gradient_descent_minibatch(X_train, Y_train,
                                                    X_dev, Y_dev,
                                                    epochs=12,
                                                    batch_size=128,
                                                    alpha=0.01,
                                                    log_every=100
)