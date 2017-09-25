# https://keon.io/deep-q-learning/
import gym
import tensorflow as tf
import numpy as np
import pylab as pl

"""
Deep Q Learning (and additional) steps

Main loop:
1. Estimate Q values based on the observed state and sample an action
2. Perform an action and observe next state
4. Save state, action, reward, next state, done (append to existing memory)
5. Randomize a mini-batch from collected experiences (memory)
6. Use randomized mini-batch to estimate target Q values and train
   Q_target = neural net output using mini-batch of states as input and then
   Q_target[action] = reward (from state) + gamma * reward (from next state)

That's it.
"""


class DQNagent():
    def __init__(self, s_size, a_size, m_size, lr):
        self.memory = []        # object:    memory
        self.history = []       # object:    training history for plotting
        self.m_size = m_size    # parameter: memory size
        self.s_size = s_size    # parameter: state size
        self.a_size = a_size    # parameter: action size
        self.lr = lr            # parameter: learning rate
        self.Q_values, self.optimization, self.sess, self.cost, self.y, self.x = self.tf_sess() # tf session

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.m_size: self.memory = self.memory[-self.m_size:]

    def act(self, state, epsilon):
        if np.random.uniform(0,1,1) < epsilon:
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

    def train(self, training_batch, discount, return_Q_targets):

        # Convert list to array and un-group stacked data
        training_batch = np.array(training_batch)
        state, action, reward, next_state, done = training_batch[:,0], training_batch[:,1], training_batch[:,2], training_batch[:,3], training_batch[:,4]

        # Calculate target Q values
        Q_target = self.sess.run(self.Q_values, feed_dict={self.x: np.stack(state)}) # Q values from state
        Q_target_next = np.amax(self.sess.run(self.Q_values, feed_dict={self.x: np.stack(next_state)}), axis=1) # Max Q values from next_state
        Game_not_done = 1 - done  # Swapping 0->1 and 1->0. Originally if done = True, but I need True if not Done.
        Q_target_one = reward + discount * Q_target_next * Game_not_done # Estimate target Q values
        for i in range(len(Q_target_one)): # Assign target Q values to appropriate action
            Q_target[i, :action[i]] = Q_target_one[i]

        # Train
        _ = self.sess.run(self.optimization, feed_dict={self.x: np.stack(state), self.y: np.stack(Q_target)})

        if return_Q_targets: return Q_target # return Q_targets if curious to see what it looks like

    def tf_sess(self):

        # Create tf placeholders and weights
        x = tf.placeholder(shape=[None, self.s_size], dtype=tf.float32)
        y = tf.placeholder(shape=[None, self.a_size], dtype=tf.float32)
        O = {'w1': tf.Variable(tf.truncated_normal(shape=[self.s_size, 64], mean=0, stddev=1/np.sqrt(self.s_size), dtype=tf.float32)),
             'w2': tf.Variable(tf.truncated_normal(shape=[64, 32], mean=0, stddev=1/np.sqrt(64), dtype=tf.float32)),
             'w3': tf.Variable(tf.truncated_normal(shape=[32, self.a_size], mean=0, stddev=1/np.sqrt(32), dtype=tf.float32))}

        # Estimate net's output
        l1 = tf.nn.relu(tf.matmul(x, O['w1']))
        l2 = tf.nn.relu(tf.matmul(l1, O['w2']))
        Q_values = tf.matmul(l2, O['w3'])

        # Estimate cost and create optimizer
        cost = tf.losses.huber_loss(y, Q_values) # tf.reduce_mean(tf.square(y - Q_values))
        optimization = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(cost)

        # Initiate tf session
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        return Q_values, optimization, sess, cost, y, x

# Hyperparameters
learning_rate = 0.0001
discount = 0.999
epsilon_max = 1
epsilon_min = 0.1 # minimum value
epsilon_d = 0.0003 # 1 - epsilon_d * games_played
memory_size = 16192 * 2
mini_batch_size = 512
train_freq = 4

# Initialize game environment, agent and else
tf.reset_default_graph()
env = gym.make('Breakout-ram-v0')
agent = DQNagent(s_size=len(env.reset()), a_size=len(env.unwrapped.get_action_meanings()), m_size=memory_size, lr=learning_rate)
pl.ioff() # to turn off plotting in IDE (save it to a file instead) pl.ion() to reverse

# Reset variables
state = env.reset() / 255
epsilon = 1
frame = 0
score = 0
running_score = 0
games_played = 0
memory_is_full = False

# Main loop
while True:

    # Chose and perform an action, save experience
    action = np.argmax(agent.act(state, epsilon))
    next_state, reward, done, _ = env.step(action)
    agent.remember(state, action, reward, next_state / 255, done)

    # Update other variables
    state = next_state / 255
    score += reward
    frame += 1
    memory_is_full = (len(agent.memory) == memory_size)

    # Sample training batch and train
    if memory_is_full and frame % train_freq == 0:
        training_batch = agent.memory_batch(mini_batch_size)
        agent.train(training_batch, discount, return_Q_targets=False)

    if done: # Stuff to do if game is finished

        # Reset, update and etc.
        state = env.reset() / 255
        if memory_is_full: epsilon = epsilon_max - epsilon_d * games_played if epsilon > epsilon_min else epsilon_min
        running_score = running_score * 0.95 + 0.05 * score
        agent.history.append(running_score)

        # Print and save
        print('Game: %d, R: %.2f, Running R: %.2f, Eps: %.2f, Memory: %d' % (games_played, score, running_score, epsilon, len(agent.memory)))
        pl.plot(agent.history)
        pl.savefig('DGQ Breakout test.png')

        # Update other variables
        frame = 0
        score = 0
        if memory_is_full: games_played += 1
