import numpy as np
import tensorflow as tf
import pylab as pl
from urllib.request import urlopen


"""
Predicting next word with single layer recurrent neural net using tensorflow.

Say we have a short story excerpt:
"Twice a day, at eleven and six, the old fellow takes his dog for a walk. For eight years that walk has never varied."

First, the story has to be reshaped into an input tensor. Suppose we decide to predict the 8'th word using last 7 words.
The input fed into tensorflow rnn should be like this:
[[Twice, a, day, at, eleven, and, six],
[a, day, at, eleven, and, six, the],
[day, at, eleven, and, six, the, old],
[at, eleven, and, six, the, old, fellow] ... ]

The output should be like this:
[[the],
[old],
[fellow],
[takes] ...
]

Of course, firstly, the words must be encoded into some kind of numerical representation.
After feeding this data into rnn, tensorflow does all the work. The rest is self-explanatory (well, kind of).
"""


def download_data(url='http://textfiles.com/stories/aircon.txt'):
    # Longer story https://ia601603.us.archive.org/3/items/CamusAlbertTheStranger/CamusAlbert-TheStranger_djvu.txt
    text = str(urlopen(url).read())
    text = text.replace('\\r', '').replace('\\n', '').replace('\\\'', '').replace('\\xe2\\x99\\xa6', '')
    return text


def get_dicts(text):

    words = ''.join(word for word in text).split()
    words_set = set(words)
    dictionary, reverse_dictionary = {}, {}
    
    for word, id in zip(words_set, range(len(words_set))):
        dictionary[word] = id
        reverse_dictionary[id] = word
        
    return dictionary, reverse_dictionary, words


def get_data(words, dictionary, time_steps):

    x_data = np.zeros(shape=(len(words) - time_steps, time_steps))
    y_data = np.zeros(shape=(len(words) - time_steps, len(set(words))))

    for data_point, sequence_end in zip(range(len(words)), range(time_steps, len(words))):
        x_data[data_point] = [dictionary[word] for word in words[data_point:sequence_end]]
        y_data[data_point, dictionary[words[data_point]]] = 1

    return x_data, y_data


def forward_prop(x, w, n_hidden):

    # single hidden layer rnn with tanh activation in hidden layer and softmax activation in the last layer
    # tf.contrib.rnn.static_rnn create weights and biases automatically, so there is no need to initiate it manually
    # to follow things up, you can check all the tf variables by tf.get_collection('variables')

    x_split = tf.split(x, time_steps, 1)

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, activation=tf.nn.tanh)  # or MultiRNNCell([cell, cell, cell])
    lstm_cell_with_dropout = tf.contrib.rnn.DropoutWrapper(lstm_cell, state_keep_prob=0.9)

    outputs, state = tf.contrib.rnn.static_rnn(lstm_cell_with_dropout, x_split, dtype=tf.float32)
    logits = tf.matmul(outputs[-1], w)  # logits are used for cross entropy
    output = tf.nn.softmax(logits)

    return logits, output


def get_mini_batch(x, y):  # single data point, in this case
    data_point = np.random.randint(len(x))
    x_batch = x[[data_point]]  # apparently, additional brackets helps to keep dim info
    y_batch = y[[data_point]]
    return x_batch, y_batch


# parameters
n_hidden = 256  # number of neurons in hidden layer
time_steps = 10  # size of sequence of words
learning_rate = 1e-3

# download and prepare data, initiate weights
text = download_data()
dictionary, reverse_dictionary, words = get_dicts(text)
x_data, y_data = get_data(words, dictionary, time_steps)

# initiate tf placeholders
x = tf.placeholder(tf.float32, [None, time_steps])
y = tf.placeholder(tf.float32, [None, len(dictionary)])

# create other tf objects
w = tf.Variable(tf.random_normal([n_hidden, len(dictionary)]), dtype=tf.float32) # last layer weights
logits, y_ = forward_prop(x, w, n_hidden)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
accuracy = tf.equal(tf.argmax(y_, axis=1), tf.argmax(y, axis=1))

# initiate tf session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

acc_t = []  # list of accuracy values over some number of iterations (True or False, for a single data point)
acc_hist = []  # list of history of average accuracies
iter = 0

# training loop
while True:

    # generate mini batch
    x_batch, y_batch = get_mini_batch(x_data, y_data)

    # train
    _, acc = sess.run([optimizer, accuracy], feed_dict={x: x_batch, y: y_batch})

    # other stuff
    acc_t.append(acc)
    iter += 1

    # plot
    if iter % 100 == 0:
        acc_hist.append(np.average(acc_t))
        pl.plot(acc_hist); pl.pause(1e-10)
        acc_t = []

# let rnn generate a story
for i in range(len(x_data)):
    
    id = np.argmax(sess.run([y_], feed_dict={x: x_data[[i]]})[0], axis=1)
    word = reverse_dictionary[int(id)]
    
    if i % 30 != 0:  # same line
        print(word, end=' ')
    else:  # new line
        print(word)
