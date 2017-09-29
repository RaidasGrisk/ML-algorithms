import numpy as np
import pylab as pl
from tensorflow.examples.tutorials.mnist import input_data # tf is used just for MNIST dataset

"""
What is in here?
Simple neural network and Adam optimization.

Short explanation to help understand how this thing is implemented.
Neural network and its components are stored in dictionaries. Keys in dicts mark layers.
For instance L is a dictionary that stores activated values of neurons.
L[1] stores values of activated neurons in 1'st hidden layer.
This same logic is applied through all the net objects that are associated with layers.

List of variables:
weights - dictionary of weights
weights_der - dictionary of derivatives of weights
biases - dictionary of biases
biases_der - dictionary of derivatives of biases
L - dictionary of activated neuron values
A - dictionary of not-activated neuron values
y_ - predictions of output (y hat)
y - true values of output
x - input

Other variables are either part of Adam or something less relevant.

Main loop:
1. Randomize a training batch of x and y.
2. Forward and backward propagate
3. Implement Adam algorithm to converge

Hope that'll help.
"""


# define functions
def activation(input):
    return input * (input > 0) # if you change this one be sure to change activation_derivative as well


def activation_derivative(input):
    return 1 * (input > 0)


def softmax(input):
    return np.exp(input) / np.sum(np.exp(input), axis=1, keepdims=True) # this activation is applied to final layer


def dropout(input, keep_prob):
    uniform_dist = np.random.uniform(low=0, high=1, size=np.shape(input))
    dropout_table = uniform_dist < keep_prob
    output = input * dropout_table / keep_prob
    return output


def get_weights(input, net_structure):

    # minor preparations
    weights = {}
    biases = {}
    _, features = np.shape(input)
    layers = np.arange(len(net_structure)) + 1

    # loop for creating weights
    for layer, neurons in zip(layers, net_structure):
        weights[layer] = np.random.uniform(-0.1, 0.1, (features, neurons))
        biases[layer] = np.random.uniform(-0.1, 0.1, (1, neurons))
        features = neurons

    return weights, biases


def forward_prop(input, weights, biases, dropout_keep_prob):

    # minor preparations
    L = {0: input}  # neuron values after activation
    A = {0: input}  # neuron values before activation
    layers = np.arange(len(weights)) + 1

    # forward propagation
    for layer in layers:
        L[layer] = np.dot(L[layer-1], weights[layer]) + biases[layer]
        A[layer] = L[layer]
        L[layer] = dropout(activation(L[layer]), dropout_keep_prob) if layer != len(layers) else softmax(L[layer])

    return L, A, L[len(layers)]


def back_prop(y_, y, L, A, weights):

    # minor preparations
    data_points = len(y)
    layers = np.arange(len(weights)) + 1

    weights_dz = {layers[-1]: y_ - y}  # z values
    weights_d = {}  # derivative of weights
    biases_d = {}  # derivative of biases

    for layer in reversed(layers):
        weights_dz[layer-1] = np.dot(weights_dz[layer], weights[layer].T) * activation_derivative(A[layer-1])
        weights_d[layer] = np.dot(L[layer-1].T, weights_dz[layer]) / data_points
        biases_d[layer] = np.sum(weights_dz[layer], axis=0) / data_points

    return weights_d, biases_d


def get_cost(y_, y):
    return np.sum(-y * np.log(y_) - (np.ones(np.shape(y)) - y) * np.log(1 - y_)) / len(y_)


def get_accuracy(y_, y):
    y_ = np.argmax(y_, axis=1)
    y = np.argmax(y, axis=1)
    accuracy = np.mean(np.equal(y_, y))
    return accuracy


def get_data_batch(x, y, batch_size):
    indexes = np.random.choice(len(y) - 1, batch_size)
    x_batch = x[indexes, :]
    y_batch = y[indexes, :]
    return x_batch, y_batch


# load data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.validation.images
y_test = mnist.validation.labels

# parameters
lr = 0.001
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-08
dropout_keep_prob = 0.9
mini_batch_size = 500
net_structure = [256, 128, 10] # output layer included, input layer not included

# initiate weights and biases
weights, biases = get_weights(x_train, net_structure)

# initiate training variables
accuracy_history = []
m_weights, m_biases, v_weights, v_biases, mc_weights, mc_biases, vc_weights, vc_biases = \
[dict.fromkeys(np.arange(len(weights))+1, 0) for _ in range(8)]  # empty dicts used in Adam, dict keys mark layers

# training loop
for i in range(1000):

    # randomize training batch
    x_batch, y_batch = get_data_batch(x_train, y_train, mini_batch_size)

    # propagate forwards and backwards
    L, A, y_ = forward_prop(x_batch, weights, biases, dropout_keep_prob)
    weights_der, biases_der = back_prop(y_, y_batch, L, A, weights)

    # Adam optimization implemented exactly as formulated in the paper on page 2:
    # https://arxiv.org/pdf/1412.6980.pdf
    for layer in weights:

        # first momentum estimate
        m_weights[layer] = beta1 * m_weights[layer] + (1 - beta1) * weights_der[layer]
        m_biases[layer] = beta1 * m_biases[layer] + (1 - beta1) * biases_der[layer]

        # second raw moment estimate
        v_weights[layer] = beta2 * v_weights[layer] + (1 - beta2) * weights_der[layer] ** 2
        v_biases[layer] = beta2 * v_biases[layer] + (1 - beta2) * biases_der[layer] ** 2

        # bias corrected first moment estimate
        mc_weights[layer] = m_weights[layer] / (1 - beta1)
        mc_biases[layer] = m_biases[layer] / (1 - beta1)

        # bias corrected second raw moment estimate
        vc_weights[layer] = v_weights[layer] / (1 - beta2)
        vc_biases[layer] = v_biases[layer] / (1 - beta2)

        # update parameters
        weights[layer] -= lr * mc_weights[layer] / (vc_weights[layer] ** 0.5 + epsilon)
        biases[layer] -= lr * mc_biases[layer] / (vc_biases[layer] ** 0.5 + epsilon)

    # print cost
    print('Cost: %.2f' % get_cost(y_, y_batch))

    # plot (beware, this brings a lot of lag)
    if i % 50 == 0:
        test_set_accuracy = get_accuracy(forward_prop(x_test, weights, biases, dropout_keep_prob=1)[2], y_test)
        train_set_accuracy = get_accuracy(forward_prop(x_train, weights, biases, dropout_keep_prob=1)[2], y_train)
        accuracy_history.append((test_set_accuracy, train_set_accuracy))
        pl.plot(accuracy_history)
        pl.title('Accuracy of train and test sets')
        pl.pause(1e-10)
