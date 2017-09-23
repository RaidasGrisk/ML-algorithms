import numpy as np
import pylab as pl
from tensorflow.examples.tutorials.mnist import input_data # tf is used just for MNIST dataset

# Hypothesis
def h(O, x):
    return 1 / (1 + np.exp(1) ** (-np.dot(x, O)))


# Cost (and derivative) function
def J(O, x, y, l, der):

    y_ = h(O, x)
    m = len(x) # number of data points

    if der == 0: # return cost
        return (1/m) * np.sum(-y * np.log(y_) - np.subtract(1, y) * np.log(1 - y_)) \
               + np.sum(np.square(l * O[1:len(O)]))

    if der == 1: # return derivative
        return (1/m) * np.dot((y_ - y).T, x).T


# Accuracy
def accuracy(y, x, O):
    return np.sum([y == (h(O, x) > 0.5)]) / len(y)


# Main

# y - array of correct answers (dependent variable)
# x - arrays of other variables (independent variables)
# O - weights (rows correspond to each column of x)

# Data (MNIST dataset from tensorflow)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x_train = mnist.train.images
y_train = mnist.train.labels[:,0].reshape((len(x_train), 1)) # picking 1's label only
x_test = mnist.test.images
y_test = mnist.test.labels[:,0].reshape((len(x_test), 1))
O = np.random.uniform(-0.01, 0.01, size=(len(x_train.T), 1))

# Gradient decent
steps = 50
alpha = 1e-3
l = 0.1
train_history = []  # list to save gradient progress

for step in range(steps):

    O = O * (1 - alpha * l / len(y_train)) - alpha * J(O, x_train, y_train, l, der=1)

    cost = J(O, x_train, y_train, l, der=0)
    acc_train, acc_test = accuracy(y_train, x_train, O), accuracy(y_test, x_test, O)
    train_history.append((cost, acc_train, acc_test))
    print(step, " :", 'cost :', cost, 'train accuracy:', acc_train, 'test accuracy: ', acc_test)

pl.plot(train_history)
