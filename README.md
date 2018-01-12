# ML-algorithms in practice

Few examples of algorithms I've tried while learning ML. Hopefully it's easy to read and understand.

1. [**Policy gradient.**](/Policy_gradient_breakout.py) First take at reinforcement learning. Basic policy gradient implementation using tensorflow and Breakout from openAI's gym. I tried to keep it as simple as possible with only minor exceptions.

2. [**Deep Q Network.**](/DQN_Breakout.py) Second take at reinforcement learning. Basic DQN implementation using tensorflow and Breakout  from openAI's gym.

2. [**Predict next character with RNN.**](/RNN_predict_next_char.py) Tensorflow RNN (LSTM) biult to predict next character. After rnn is trained, it can generate new text based on some initial text input.

# Things implemented simply

The idea is to make everything simple and readable. Take a concept and implement it from scratch (numpy in this case). If you manage to do that, perhaps that will help you to understand a thing or two about the concept.

1. [**Multivariate regression.**](/Multivariate_regression.py) Optimized using gradient decent with regularization (lambda).

2. [**Logistic regression.**](/Logistic_regression.py) The same as before. Simple logistic regression. 

3. [**Vanilla neural network.**](/Neural_network_v2.py) Multi-layer perceptron optimized using Adam, regularized using dropout.

4. [**Convolutional neural network (not finished, still working on it).**](/CNN.py)

5. [**Recurrent neural network.**](/RNN_v1.py) Single hidden layer recurrent network used for MNIST clasification.

# Gifs to look at

| Generating text with LSTM |
|---------------------------|
|![](/gifs/LSTM-text-gen.gif)|

| Playing Breakout: vanilla neural net and policy gradient |
|------------------|
| ![](/gifs/Breakout.gif) |
