# http://cs229.stanford.edu/proj2016/report/tripathi-deep-reinforcement-learning-for-atari-games-report.pdf
# https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
# https://gist.github.com/greydanus/5036f784eec2036252e1990da21eda18
import numpy as np
import gym
import tensorflow as tf
import pylab as pl

# functions
def plot_pixels(observation):
    pl.imshow(observation.reshape(50,72)) # whole space 80, 120 ; not including bricks 72, 50

def prep_observation(observation, zeros_and_ones):
    obs_2d = observation[:, :, 0] # from RGB to R
    obs_2d_cut = obs_2d[93:193, 8:152] # Specific to Breakout: whole space 33:193, 8:152 ; not including bricks 93:193, 8:152
    obs_2d_cut_ds = obs_2d_cut[::2, ::2] # downsample by 2: a b c d e f d - to - a c e d
    if zeros_and_ones: obs_2d_cut_ds[obs_2d_cut_ds != 0] = 1 # convert to 0's and 1's
    obs_1d = np.ndarray.flatten(obs_2d_cut_ds, order='C') # array to list
    return obs_1d

def forward_propagation(input, O):
    # specify NN
    # in this case: 1 hidden layer with relu; softmax output
    l1 = tf.nn.relu(tf.matmul(input, O['w1']))
    l2 = tf.nn.softmax(tf.matmul(l1, O['w2']))
    return l2

def discount_and_normalize(input, discount_const, normalized, frame_shift):

    # Discounting
    data_disc = np.zeros_like(input).astype(float)
    data_disc[len(input) - 1] = input[len(input) - 1] # no discount for last reward
    for i in reversed(range(0, len(data_disc) - 1)):
        if input[i] != 0:
            data_disc[i] = input[i]  # keep original if non zero
        else:
            data_disc[i] = data_disc[i + 1] * discount_const  # this value = the value before * discount

    # Normalizing
    if normalized:
        data_disc = (data_disc - np.mean(data_disc)) / (np.std(data_disc) + 1e-10)

    # shifting to account for late reward after good actions (lame way to try solve credit assignment problem)
    if frame_shift:
        length = len(data_disc)
        data_disc_shifted = np.zeros_like(data_disc)
        data_disc_shifted[0:length-frame_shift, ] = data_disc[-(length-frame_shift):, ] # shift by x frames
        data_disc = data_disc_shifted # rename for easy return

    return data_disc

# hyperparameters
learning_rate = 0.002
reward_shift = 10 # to account for lagging reward (frames)
reward_discount = 0.99
obs_discount = 0.8 # discount last frame to account for velocity of objects (used in running_frame)
training_batch_size = 5 # number of games to perform one optimization step
neurons = 32
input_size = 3600 # hand written number if pixel observations are altered
actions = [0, 1, 2] # true actions are 1, 2, 3 this is used for index
session_name = 'LR %.4f, RS %d, RD %.2f, TB %d, N %d' % (learning_rate, reward_shift, reward_discount, training_batch_size, neurons) # filename for saving

# initialize tf placeholders
x = tf.placeholder(shape=[None, input_size], dtype=tf.float32)
y = tf.placeholder(shape=[None, len(actions)], dtype=tf.float32)
gradient_loss = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# initialize weights using xavier initialization:
# http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization
# https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/
O = {'w1': tf.Variable(tf.truncated_normal(shape=[input_size, neurons], mean=0, stddev=1/np.sqrt(input_size), dtype=tf.float32)),
     'w2': tf.Variable(tf.truncated_normal(shape=[neurons, len(actions)], mean=0, stddev=1/np.sqrt(neurons), dtype=tf.float32))}

# Trained_weighs = np.load('session_name' + '.npy').item()
# O['w1'] = tf.Variable(Trained_weighs['w1'])
# O['w2'] = tf.Variable(Trained_weighs['w2'])

# initialize other variables
x_batch, y_batch, reward_batch, games_played, running_reward, running_reward_hist = [], [], [], 0, None, []
pl.ioff() # to turn off plotting in IDE (save it to a file instead) plt.ion() to reverse
yprob = forward_propagation(x, O)
cost = tf.nn.l2_loss(y - yprob)

# initialize optimizer
# https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
gradients = optimizer.compute_gradients(loss=cost, var_list=tf.trainable_variables(), grad_loss=gradient_loss) # gradients computed with altered last layer errors (using rewards)
train_step = optimizer.apply_gradients(gradients)

# initialize gamespace
env = gym.make("Breakout-v0")
observation = env.reset()
running_observation = prep_observation(observation, zeros_and_ones=True)
# plot_pixels(prep_observation(observation)) # see how single frame looks like

# initialize tf session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# tf.reset_default_graph()

# playing / learning loop
while True:
    # env.render()

    # preprocess observation
    modified_observation = prep_observation(observation, zeros_and_ones=True) # alter image: resize, flatten etc.

    # create running frame (input to nn)
    running_observation = modified_observation + running_observation * obs_discount
    running_observation[running_observation < 0.5] = 0 ; running_observation[running_observation > 1.0] = 1
    # plot_pixels(running_observation) ; pl.pause(1e-10)

    # sample a policy from the network (using running frame made from preprocessed observations)
    output = sess.run(yprob, feed_dict = {x: np.stack([running_observation])})[0]
    action = np.random.choice(a=len(actions), p=output)

    # create 'correct' nn output (the correct one is always the one that was sampled)
    ylabel = np.zeros_like(output) ; ylabel[action] = 1

    # step the environment and get new observation
    observation, reward, done, info = env.step(action + 1) # +1 for actions 1, 2, 3 instead of 0, 1, 2

    # record game history
    x_batch.append(running_observation)
    y_batch.append(ylabel)
    reward_batch.append(reward)

    if done: # end of collecting data for 1 game

        # reset game and observations
        observation = env.reset() ; running_observation = prep_observation(observation, zeros_and_ones=True)
        games_played += 1

        if games_played % training_batch_size == 0: # perform 1 optimization step after playing x games (signle batch)

            # optimization
            gradient_loss_batch = discount_and_normalize(reward_batch, discount_const=reward_discount, normalized=True, frame_shift=reward_shift)
            sess.run([train_step], feed_dict={x: np.stack(x_batch),
                                              y: np.stack(y_batch),
                                              gradient_loss: np.stack([gradient_loss_batch]).T})

            # print intermediate results
            average_reward = np.divide(np.sum(reward_batch), training_batch_size) # average reward over a batch of games
            running_reward = 0.95 * running_reward + 0.05 * average_reward if running_reward is not None else average_reward
            running_reward_hist.append(running_reward)
            print('Running reward: %.2f Average batch reward: %.2f Games played: %d' % (running_reward, average_reward, games_played))

            # reset batch memory and save
            x_batch, y_batch, reward_batch = [], [], []

            # save stuff
            pl.plot(running_reward_hist)
            pl.savefig('Running_reward_hist' + session_name + '.png')
            np.save('Breakout_v1_trained_weights' + session_name, sess.run(O))
