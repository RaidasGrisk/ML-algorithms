import numpy as np
import tensorflow as tf
import gym
import pylab as pl

""""
Basic explanation of Policy Gradient:

In policy gradient, neural net outputs a probability for each available action.
The idea of this method is to train neural net so that probabilities of actions
after which you've gotten reward in your past experiences are increased.

For example, the player is in state 'A' (pixel values in this example)
and he randomly picked action 3 from [1, 2, 3] set of actions.
He got lucky and this action yielded him a reward of 1! Wohoo!
We've collected this data and now we'll train the network to increase the
probability of action 3 whatever we are in a state similar or equal to 'A'.

That's the basic idea. Here's a short overview of the main loop.

First you play the game :
1. Pre-process state (observation) and be ready to feed it to neural net.
2. Input the observation into neural net and sample an action from its output.
   By sample, I mean sample an action from action set [1, 2, 3] when the
   probabilities of actions are e.g. [0.2, 0.3, 0.5]. Simple stochastic process.
3. Make that action and observe the results: state, reward, ...
4. Create 'correct' output value (ylabel) of neural net for this state.
   This step is a little tricky to understand at first. The 'correct' output is
   always the one you've sampled from neural net. E.g if action 3 was correct,
   than correct output is [0, 0, 1] where third column corresponds to action 3.
4. Save the data: state, the 'correct' output and the reward you've got.

This is what you do after each frame. Now do this for a while, a few 
(or one, however you like) games to collect enough data for training.

After you've collected enough data, do one iteration of training:
1. Construct gradient loss vector. This step is the core of gradient decent method.
   It is simply a vector of discounted reward vector values collected during the game.
   You can make additional transformations, e.g. normalize or/and shift this vector to improve results.
2. Adjust the errors of net's final layer. Before backprop, you multiply the errors of
   your network's final layer with this vector of discounted reward. In turn, this will
   increase the errors before sequences of actions that led to reward.
3. Do optimization. After adjusting errors and performing optimization using backprop, the
   probability of a sequence of actions which led to reward will increase.
4. Clear all the data that you've just trained with and play the game again to collect new data.

That's it.

This code is highly based on the work of these guys (it helped me to understand Policy Gradient big time):
https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
https://gist.github.com/greydanus/5036f784eec2036252e1990da21eda18
"""

# functions
def prep_observation(observation, zeros_and_ones):
    obs_2d = observation[:, :, 0] # from RGB to R
    obs_2d_cut = obs_2d[93:193, 8:152] # Specific to Breakout: whole space 33:193, 8:152 ; not including bricks 93:193, 8:152
    obs_2d_cut_ds = obs_2d_cut[::2, ::2] # downsample by 2: a b c d e f d -> to -> a c e d
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
        if input[i] != 0: # keep original if non zero
            data_disc[i] = input[i]  
        else: # this value = the value before * discount
            data_disc[i] = data_disc[i + 1] * discount_const  

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


def plot_pixels(observation): # this one is used to plot how input to nn looks like
    pl.imshow(observation.reshape(50,72)) # whole space 80, 120 ; not including bricks 50, 72


# hyperparameters
learning_rate = 0.002
reward_shift = 10 # to account for lagging reward (frames)
reward_discount = 0.99
obs_discount = 0.8 # discount last frame to account for velocity of objects (used in running_frame)
training_batch_size = 5 # number of games to perform one optimization step
neurons = 32 # single hidden layer with that many neurons
input_size = 3600 # hand written number indicating number of pixels fed into nn
actions = [0, 1, 2] # true actions are 1, 2, 3 this is used for indexing
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

# Run this to load old weights if resuming training
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
gradients = optimizer.compute_gradients(loss=cost, var_list=tf.trainable_variables(), grad_loss=gradient_loss) # gradients computed with adjusted last layer's errors (using rewards)
train_step = optimizer.apply_gradients(gradients)

# initialize gamespace
env = gym.make("Breakout-v0")
observation = env.reset()
running_observation = prep_observation(observation, zeros_and_ones=True)
# plot_pixels(prep_observation(observation)) # run this to see how single frame fed to nn looks like

# initialize tf session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# tf.reset_default_graph() # run this to clear all global variables

# playing / learning loop
while True:
    # env.render()

    # preprocess observation
    modified_observation = prep_observation(observation, zeros_and_ones=True) # alter image: resize, flatten etc.

    # create running frame (input to nn)
    running_observation = modified_observation + running_observation * obs_discount
    running_observation[running_observation < 0.5] = 0 ; running_observation[running_observation > 1.0] = 1 # trim
    # plot_pixels(running_observation) ; pl.pause(1e-10) # run this to plot running frame each frame. Beware of lag.

    # sample a policy from the network (using running frame made from preprocessed observations)
    output = sess.run(yprob, feed_dict = {x: np.stack([running_observation])})[0]
    action = np.random.choice(a=len(actions), p=output)

    # step the environment and get new observation
    observation, reward, done, info = env.step(action + 1) # +1 for actions 1, 2, 3 instead of 0, 1, 2

    # create 'correct' nn output (the correct one is always the one that was sampled)
    ylabel = np.zeros_like(output) ; ylabel[action] = 1

    # record game history
    x_batch.append(running_observation)
    y_batch.append(ylabel)
    reward_batch.append(reward)

    if done: # end of collecting data for 1 game

        # reset game and observations
        observation = env.reset() ; running_observation = prep_observation(observation, zeros_and_ones=True)
        games_played += 1

        if games_played % training_batch_size == 0: # perform 1 optimization step after playing x games (single batch)

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
            pl.savefig('Running_reward_hist' + session_name + '.png') # look at the progress by opening this file
            # np.save('Breakout_v1_trained_weights' + session_name, sess.run(O))
