import numpy as np
import pylab as pl
from tensorflow.examples.tutorials.mnist import input_data # tf is used just for MNIST dataset

"""
Some explanations:
1. N, C, H, W -> Nr of data-points, Channels, Height, Width

Credits for im2col and col2im (i's a massive headache, so I borrowed it) goes to:
https://github.com/shenxudeu/Convnet/blob/master/assignment3/cs231n/im2col.py#L14
"""


# functions
def relu(input):
    return input * (input > 0)


def relu_derivative(input):
    return 1 * (input > 0)


def softmax(input):
    return np.exp(input) / np.sum(np.exp(input), axis=1, keepdims=True) # this activation is applied to final layer


def get_padding(weights, stride):

    """
    Function returns padding such that convolution output height and width
    is input height and width divided by stride.

    Function returns a tuple of pad parameters, where first parameter is corresponds to
    left side or top, second to right side and bottom padding.
    """

    # minor preparations
    _, _, h, _ = np.shape(weights)
    pad_size = h - stride

    if pad_size == 0:  # if there is no need for padding

        return 0, 0

    else:  # if padding is needed

        if pad_size % 2 != 0:  # if pad is not symmetric
            end_pad = int(pad_size / 2)
            front_pad = pad_size - end_pad

        else:  # if pad is symmetric
            end_pad = pad_size / 2
            front_pad = pad_size / 2

        return int(front_pad), int(end_pad)


def get_im2col_indices(input, weights, padding, stride):

    """
    Function returns indices for im2col operation
    Credits: https://github.com/shenxudeu/Convnet/blob/master/assignment3/cs231n/im2col.py#L14
    """

    n, c, h, w = np.shape(input)
    pad_h = np.sum(padding)
    pad_w = np.sum(padding)

    _, _, _, field_height = np.shape(weights)
    _, _, field_width, _ = np.shape(weights)

    out_height = int((h + pad_h - field_height) / stride + 1)
    out_width = int((w + pad_w - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, c)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * c)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(c), field_height * field_width).reshape(-1, 1)

    return (k, i, j)


def im2col_indices(input, weights, padding, stride):

    """
    Implements im2col based on indices returned from get_im2col_indices
    Credits: https://github.com/shenxudeu/Convnet/blob/master/assignment3/cs231n/im2col.py#L14
    With minor modifications.
    """

    _, _, _, field_height = np.shape(weights)
    _, _, field_width, _ = np.shape(weights)
    _, c, _, _ = input.shape

    x_padded = np.pad(input, ((0, 0), (0, 0), padding, padding), mode='constant')

    k, i, j = get_im2col_indices(input, weights, padding, stride)

    cols = x_padded[:, k, i, j]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * c, -1)

    return cols


def convolve_forward_once(input, weights, stride, padding):

    """
    Implements a single convolution using get_im2col_indices and im2col_indices
    Credits: https://github.com/shenxudeu/Convnet/blob/master/assignment3/cs231n/im2col.py#L14
    Heavily modified.
    """

    # minor preparations
    n, c, h, _ = np.shape(input)
    _, filter_num, filter_height, filter_width = np.shape(weights)

    output_height = int((h + np.sum(padding) - filter_width) / stride + 1)
    output_width = int((h + np.sum(padding) - filter_height) / stride + 1)

    # im2col and forward pass
    input_col = im2col_indices(input, weights, padding, stride)
    weights_prep = np.tile(weights.reshape((filter_num, -1)), c)           # Modification
    output_res = np.dot(weights_prep, input_col)                           # Modification
    # output_res = np.dot(weights.reshape((filter_num, -1)), input_col)    # Original line

    # reshaping output
    output = output_res.reshape(filter_num, output_height, output_width, n)
    output = output.transpose(3, 0, 1, 2)

    return output


def forward_prop_net(input, weights_cn, weights_fc, stride):

    # forward prop convolution layers
    L = {0: input}
    A = {0: input}

    for layer in weights_cn.keys():

        padding = get_padding(weights_cn[layer], stride)
        L[layer] = convolve_forward_once(L[layer - 1], weights_cn[layer], stride, padding)
        A[layer] = L[layer]
        L[layer] = relu(L[layer])

    # forward prop fully connected layers
    n, c, h, w = np.shape(L[len(L) - 1])
    first_layer = sorted(weights_fc.keys())[0]
    last_layer = sorted(weights_fc.keys())[-1]

    for layer in weights_fc.keys():

        L[layer] = np.dot(L[layer-1] if layer != first_layer else L[layer-1].reshape(n, c * h * w), weights_fc[layer])
        A[layer] = L[layer]
        L[layer] = relu(L[layer]) if layer != last_layer else softmax(L[layer])

    return L, A


# data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
all_images = mnist.train.images
images = all_images.reshape((55000, 1, 28, 28))[0:1000 :, :, :]


# initialize weights
# weights_cn - weights of convolutions
# weights_fc - weights of fully connected layers
# _, filter_num, height, width
weights_cn = {1: np.random.uniform(-1, 1, (1, 4, 3, 3)),
              2: np.random.uniform(-1, 1, (1, 8, 3, 3)),
              3: np.random.uniform(-1, 1, (1, 12, 3, 3))}

weights_fc = {4: np.random.uniform(-1, 1, (108, 64)), # 108 is computed manually
              5: np.random.uniform(-1, 1, (64, 10))}


# main loop
for _ in range(100):

    # forward and backward propagate
    L, A = forward_prop_net(images, weights_cn, weights_fc, stride=2)


# plot convolutions
pl.imshow(A[0][2, 0, :, :])
pl.imshow(A[1][2, 0, :, :])
pl.imshow(A[2][2, 0, :, :])
pl.imshow(A[3][2, 0, :, :])
