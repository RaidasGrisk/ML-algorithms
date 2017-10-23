"""
Predicting next character with RNN using Tensorflow.
Suppose we decide to predict the 11'th character using the last 10 characters.

Say we have a short story excerpt:
"before the midnight..."

The tensor fed into tensorflow's rnn should be shaped like this [some nr of rows, 10]:
[
[b, e, f, o, r, e,  , t, h, e],
[e, f, o, r, e,  , t, h, e,  ],
[f, o, r, e,  , t, h, e,  , m],
[o, r, e,  , t, h, e,  , m, i],
[r, e,  , t, h, e,  , m, i, d], ... ]

The output should be like this (the character we want to predict):
[
[ ],
[m],
[i],
[d],
[n], ... ]

Roughly speaking, rnn does this process: output(t) = input_column(t) * w1 + hidden_state(t-1) * w2.
So, before returning the final output (i.e. the 11th character) the net will repeat this process 10 times.

Of course, firstly, characters must be encoded into some kind of numerical representation.
After feeding this data into rnn, tensorflow does all the work. The rest is self-explanatory (well, kind of).

Helpful source: https://medium.com/towards-data-science/lstm-by-example-using-tensorflow-feb0c1968537
"""

import numpy as np
import tensorflow as tf
import pylab as pl
from urllib.request import urlopen


def download_data(url='https://ia601603.us.archive.org/3/items/CamusAlbertTheStranger/CamusAlbert-TheStranger_djvu.txt'):
    # Short story e.g.: http://textfiles.com/stories/aircon.txt
    # Longer story e.g.: https://ia601603.us.archive.org/3/items/CamusAlbertTheStranger/CamusAlbert-TheStranger_djvu.txt
    text = str(urlopen(url).read())
    text = text.replace('\\r', '')\
               .replace('\\n', '')\
               .replace('\\\'', '')\
               .replace('\\xe2\\x99\\xa6', '')\
               .replace('\\xe2\\x80\\x94', ' ')
    return text


def get_dicts(text):

    """
    Returns a tuple of three objects:
    dictionary is a dictionary that contains all unique characters in given text (keys) and their ids (values)
    reverse_dictionary is a dictionary that contains character's ids (as keys) and characters (as values)
    chars is text converted into a list, where each element is single character.
    """
    
    chars = list(text)  # splits strings into chars and puts it into a list
    # chars = ''.join(char for char in text).split()  # splits string into swords and stores it in a list

    dictionary, reverse_dictionary = {}, {}
    for id, char in enumerate(set(chars)):
        dictionary[char] = id
        reverse_dictionary[id] = char
    
    return dictionary, reverse_dictionary, chars


def get_data(chars, dictionary, time_steps):

    """
    Returns data ready to be fed into neural net:
    x_data contains all sequences of characters (not chars, but their ids!). Single row corresponds to single sequence.
    y_data contains the id of next character in a sequence
    """
    
    x_data = np.zeros(shape=(len(chars) - time_steps, time_steps))
    y_data = np.zeros(shape=(len(chars) - time_steps, len(set(chars))))

    for data_point, sequence_end in zip(range(len(chars)), range(time_steps, len(chars))):
        x_data[data_point] = [dictionary[char] for char in chars[data_point:sequence_end]]
        y_data[data_point, dictionary[chars[sequence_end]]] = 1

    return x_data, y_data


def forward_prop(x, w, n_hidden):

    """
    RNN with tanh activation in hidden layers and softmax activation in the last layer.
    Number of elements in n_hidden correspond to layers, each number corresponds to number of neurons in a layer.
    tf.contrib.rnn.static_rnn create weights and biases automatically, so there is no need to initiate it manually
    to follow things up, you can check all the tf variables by tf.get_collection('variables')
    """
    
    # split the data to time_steps columns, to recure one column by another
    x_split = tf.split(x, time_steps, 1)

    # stack lstm cells, a cell per hidden layer
    stacked_lstm_cells = []  # a list of lstm cells to be inputed into MultiRNNCell
    for layer_size in n_hidden:
        stacked_lstm_cells.append(tf.contrib.rnn.BasicLSTMCell(layer_size, activation=tf.nn.tanh))

    # create the net and add dropout
    lstm_cell = tf.contrib.rnn.MultiRNNCell(stacked_lstm_cells)
    lstm_cell_with_dropout = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=0.9)

    # forwawrd propagate
    outputs, state = tf.contrib.rnn.static_rnn(lstm_cell_with_dropout, x_split, dtype=tf.float32)
    logits = tf.matmul(outputs[-1], w)  # logits are used for cross entropy
    output = tf.nn.softmax(logits)

    return logits, output


def get_mini_batch(x, y, batch_size):
    data_point = np.random.randint(0, len(x), size=batch_size)
    x_batch = x[data_point]
    y_batch = y[data_point]
    return x_batch, y_batch


def generate_new_text(txt, print_length, new_line, dictionary, reverse_dictionary):

    """
    Generates text by predicting next character.
    Function arguments:
    txt - some text to start with. Must be time_steps in length.
    print_length - the length of predicted text in characters
    new_line - after that many characters it prints in a new line
    """

    # prepare text to feed it into net
    txt_sample = list(txt)  # do this if trying to predict next char
    # txt_sample = ''.join(word for word in txt).split()  # do this if trying to predict next word
    x_data_sample = np.zeros(shape=(1, time_steps))

    for id, char in zip(range(len(txt_sample)), txt_sample):
        x_data_sample[:, id] = dictionary[char]

    # print the text given as argument
    print(txt)

    # predict next char, print, use predicted char to predict next and so on
    txt_length = 1
    for _ in range(print_length):
        next_char_id = np.argmax(sess.run([y_], feed_dict={x: x_data_sample,})[0], axis=1)
        next_char = reverse_dictionary[next_char_id[0]]
        x_data_sample = np.delete(x_data_sample, 0, axis=1)
        x_data_sample = np.insert(x_data_sample, len(x_data_sample[0]), next_char_id, axis=1)

        if txt_length % new_line != 0:  # same line
            print(next_char, end='')
        else:  # new line
            print(next_char)

        txt_length += 1


# hyper-parameters
n_hidden = [74, 126]  # neurons in a layers, first item corresponds to first layer and so on
batch_size = 250
time_steps = 40  # size of sequence of words
learning_rate = 1e-3

# download and prepare data, initiate weights
text = download_data()
dictionary, reverse_dictionary, chars = get_dicts(text)
x_data, y_data = get_data(chars, dictionary, time_steps)

# initiate tf placeholders
x = tf.placeholder(tf.float32, [None, time_steps])
y = tf.placeholder(tf.float32, [None, len(dictionary)])

# create other tf objects
w = tf.Variable(tf.random_normal([n_hidden[-1], len(dictionary)]), dtype=tf.float32) # last layer weights
logits, y_ = forward_prop(x, w, n_hidden)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_, axis=1), tf.argmax(y, axis=1)), tf.float32))

# initiate tf session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# initiate new training session
accuracy_hist = []
iter = 0

# training loop
while True:

    # get mini batch and train
    x_batch, y_batch = get_mini_batch(x_data, y_data, batch_size)
    _ = sess.run([optimizer], feed_dict={x: x_batch, y: y_batch})

    # other stuff
    iter += 1

    # plot and print
    if iter % 50 == 0:
        error, acc = sess.run([cost, accuracy], feed_dict={x: x_batch, y: y_batch})
        accuracy_hist.append(acc)
        pl.plot(accuracy_hist); pl.pause(1e-99)
        print('Cost: %.2f' % error)

# generate new text by giving rnn something to start
starting_txt = 'Ive had the body moved to our little mor'
generate_new_text(txt=starting_txt, print_length=1000, new_line=100, dictionary=dictionary, reverse_dictionary=reverse_dictionary)

