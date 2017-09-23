# https://keon.io/deep-q-learning/
import gym
import tensorflow as tf
import numpy as np
import pylab as pl

"""
Deep Q Learning (and additional) steps

Collect data for training:
1. Estimate Q values based on the observed state and chose an action
2. Perform an action and observe next state
4. Save state, action, reward, next state, done (append to existing memory)
5. Randomize a mini-batch from the experiences gathered (memory)
6. Train using the randomized mini-batch                           -

Train neural net:
1. Estimate target Q values: Q_target[action] = reward (from state) + gamma * reward (from next state)
2. Train using target Q values as correct output of NN
"""


class DQNagent():
    def __init__(self, s_size, a_size, m_size, gamma, eps, eps_d, lr):
        self.memory = []        # object:    memory
        self.history = []       # object:    training history for plotting
        self.m_size = m_size    # parameter: memory size
        self.s_size = s_size    # parameter: state size
        self.a_size = a_size    # parameter: action size
        self.eps = eps          # parameter: random action probability in the beginning (explore-exploit dilemma)
        self.eps_d = eps_d      # parameter: decay of probability of random action
        self.gamma = gamma      # parameter: discount for next reward
        self.lr = lr            # parameter: learning rate
        self.Q_values, self.optimization, self.sess, self.cost, self.y, self.x = self.tf_sess() # tf session

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.m_size: self.memory = self.memory[-self.m_size:]

    def act(self, state):
        if np.random.uniform(0,1,1) < self.eps:
            action = np.zeros(self.a_size)
            action[np.random.randint(0,self.a_size,1)] = 1
        else:
            action = self.sess.run(self.Q_values, feed_dict={self.x: [state]})
        return action

    def memory_batch(self, batch_size):
        if len(self.memory) > batch_size:
            batch_id = np.random.choice(len(self.memory) - 1, batch_size)
            mini_batch = [self.memory[index] for index in batch_id]
        else:
            mini_batch = self.memory
        return mini_batch

    def tf_sess(self):
        x = tf.placeholder(shape=[None, self.s_size], dtype=tf.float32)
        y = tf.placeholder(shape=[None, self.a_size], dtype=tf.float32)
        O = {'w1': tf.Variable(tf.truncated_normal(shape=[self.s_size, 128], mean=0, stddev=0.01, dtype=tf.float32)),
             'w2': tf.Variable( tf.truncated_normal(shape=[128, self.a_size], mean=0, stddev=0.01, dtype=tf.float32))}
        Q_values = tf.matmul(tf.nn.relu(tf.matmul(x, O['w1'])), O['w2'])
        cost = tf.reduce_mean(tf.square(y - Q_values))
        optimization = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(cost)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        return Q_values, optimization, sess, cost, y, x

    def train(self, training_batch, return_Q_targets):

        # convert appended lists into numpy arrays
        training_batch = np.array(training_batch)

        # Ungroup stacked data
        state, action, reward, next_state, done = training_batch[:,0], training_batch[:,1], training_batch[:,2], training_batch[:,3], training_batch[:,4]

        # Estimate predictions of Q_values from state
        Q_target = self.sess.run(self.Q_values, feed_dict={self.x: np.stack(state)})

        # Estimate predictions of Q_values from next_state
        Q_target_next = np.amax(self.sess.run(self.Q_values, feed_dict={self.x: np.stack(next_state)}), axis=1)

        # Swapping 0->1 and 1->0
        Q_target_not_done = 1 - done  # Swapping values. Originally, if done = True, I need True if not Done.

        # Estimate target values of Q_value
        Q_target_one = reward + 0.95 * Q_target_next * Q_target_not_done

        # Assign target Q values to appropriate action
        for i in range(len(Q_target_one)):
            Q_target[i, :action[i]] = Q_target_one[i]

        # Train
        _ = self.sess.run(self.optimization, feed_dict={self.x: np.stack(state), self.y: np.stack(Q_target)})

        if return_Q_targets: return Q_target

tf.reset_default_graph()
env = gym.make('Breakout-ram-v0')
agent = DQNagent(s_size=len(env.reset()), a_size=len(env.unwrapped.get_action_meanings()), m_size=16192, gamma=0.99, eps=1, eps_d=0.99955, lr=0.0002)
pl.ioff() # to turn off plotting in IDE (save it to a file instead) pl.ion() to reverse

state = env.reset()
score = 0
game = 0
running_score = 0

while True:

    # Chose and performe an action, save experience
    action = np.argmax(agent.act(state))
    next_state, reward, done, _ = env.step(action)
    agent.remember(state, action, reward, next_state, done)

    # Sample training batch and train
    training_batch = agent.memory_batch(256)
    agent.train(training_batch, return_Q_targets=False)

    # Other
    state = next_state
    score += reward

    if done:

        # Reset and etc.
        env.reset()
        agent.eps *= agent.eps_d
        running_score = running_score * 0.95 + 0.05 * score
        agent.history.append(running_score)

        # Print and save
        print('Game: %d, Reward: %.2f, Eps: %.2f, Memory: %d' % (game, running_score, agent.eps, len(agent.memory)))
        pl.plot(agent.history)
        pl.savefig('DGQ Breakout test.png')

        # Other
        score = 0
        game += 1

# outside loop for epsilon, when performing an action instead of inside
# epsilon decay: epsilon = 1 - games * X
# if memory is not full, do not train
