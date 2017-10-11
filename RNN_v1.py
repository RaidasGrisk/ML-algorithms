import numpy as np
import pylab as pl
from tensorflow.examples.tutorials.mnist import input_data # tf is used just for MNIST dataset


def activation(input):
    return input * (input > 0) # if you change this one be sure to change activation_derivative as well


def activation_derivative(input):
    return 1 * (input > 0)


def softmax(input):
    return np.exp(input) / np.sum(np.exp(input), axis=1, keepdims=True) # this activation is applied to final layer


def forward_prop(x, y, w1, w2, rw1):

    _, time_steps, _ = np.shape(x)
    a1_values = [np.zeros((len(x), len(w1.T)))]
    a2_values = []
    a2_deltas = []

    for time_step in range(time_steps):
        a1 = activation(np.dot(x[:, time_step, :], w1) + np.dot(a1_values[-1], rw1))
        a2 = softmax(np.dot(a1, w2))
        a2_deltas.append(a2 - y)
        a1_values.append(a1)
        a2_values.append(a2)

    return a1_values, a2_values, a2_deltas, a2_values[-1]


def backward_prop(x, w1, w2, rw1, a1_values, a2_deltas):

    """
    Credits: https://github.com/km1414/Deep-stuff/blob/master/Neural%20networks/RNN_np.py
    I've modified parts of it.
    """

    data_points, time_steps, _ = np.shape(x)
    next_a1_delta = np.zeros_like(a1_values[-1])  # np.zeros((len(x), len(w1.T)))
    dw2 = 0
    dw1 = 0
    drw1 = 0

    for time_step in reversed(range(time_steps)):

        a0 = x[:, time_step, :]
        a1_previous = a1_values[time_step]
        a1 = a1_values[time_step + 1]
        a2_delta = a2_deltas[time_step]
        a1_delta = (np.dot(next_a1_delta, rw1.T) + np.dot(a2_delta, w2.T)) * activation_derivative(a1)

        dw1 += np.dot(a0.T, a1_delta) / data_points
        dw2 += np.dot(a1.T, a2_delta) / data_points
        drw1 += np.dot(a1_previous.T, a1_delta) / data_points

        next_a1_delta = a1_delta

    return dw1, dw2, drw1


def get_cost(y_, y):
    return np.sum(-y * np.log(y_) - (np.ones(np.shape(y)) - y) * np.log(1 - y_)) / len(y_)


def get_accuracy(y_, y):
    y_ = np.argmax(y_, axis=1)
    y = np.argmax(y, axis=1)
    accuracy = np.mean(np.equal(y_, y))
    return accuracy


def one_adam_step(parameter, gradient, cache, beta1=0.9, beta2=0.999, alpha=0.001, epsilon=1e-8):

    m, v = cache

    # Adam optimization implemented exactly as formulated in the paper on page 2:
    # https://arxiv.org/pdf/1412.6980.pdf

    m = beta1 * m + (1 - beta1) * gradient  # update biased first moment estimate
    v = beta2 * v + (1 - beta2) * gradient ** 2  # update biased second raw moment estimate
    m_ = m / (1 - beta1)  # compute bias-corrected first moment estimate
    v_ = v / (1 - beta2)  # compute bias-corrected second raw moment estimate
    updated_parameter = parameter - alpha * m_ / (v_ ** 0.5 + epsilon)  # update parameters

    return updated_parameter, (m, v)


def get_data_batch(x, y, batch_size):
    indexes = np.random.choice(len(y) - (batch_size+1))
    x_batch = x[indexes:indexes+batch_size, :, :]
    y_batch = y[indexes:indexes+batch_size, :]
    return x_batch, y_batch


# hyper-parameters
time_steps = 4  # an image is separated into time_steps parts for recurrence
batch_size = 200

# load data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x_train = mnist.train.images.reshape(-1, time_steps, 784//time_steps)  # shape=(data_points, time_step, pixel/time_step)
y_train = mnist.train.labels
x_test = mnist.validation.images.reshape(-1, time_steps, 784//time_steps)
y_test = mnist.validation.labels

# reset training
w1 = np.random.uniform(-0.1, 0.1, (784//time_steps, 50))
w2 = np.random.uniform(-0.1, 0.1, (50, 10))
rw1 = np.random.uniform(-0.1, 0.1, (50, 50))
w1_cache, w2_cache, rw1_cache = (0, 0), (0, 0), (0, 0)  # adam cache
accuracy_hist = []

# training loop
for i in range(100):

    # randomize a training batch
    x_batch, y_batch = get_data_batch(x_train, y_train, batch_size)

    # forward and backward prop
    a1_values, a2_values, a2_deltas, y_ = forward_prop(x_batch, y_batch, w1, w2, rw1)
    dw1, dw2, drw1 = backward_prop(x_batch, w1, w2, rw1, a1_values, a2_deltas)

    # update parameters
    w1, w1_cache = one_adam_step(parameter=w1, gradient=dw1, cache=w1_cache)
    w2, w2_cache = one_adam_step(parameter=w2, gradient=dw2, cache=w2_cache)
    rw1, rw1_cache = one_adam_step(parameter=rw1, gradient=drw1, cache=rw1_cache)

    # plot and print
    accuracy_hist.append(get_accuracy(forward_prop(x_test, y_test, w1, w2, rw1)[3], y_test))
    pl.plot(accuracy_hist); pl.pause(1e-10)
    print(get_cost(y_, y_batch), get_accuracy(y_, y_batch))

