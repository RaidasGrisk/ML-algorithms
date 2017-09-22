import numpy as np
import pylab as pl


# Hypothesis
def h(O, x):
    return np.dot(x, O)


# Cost (and derivative) function
def J(O, x, y, l, der):

    m = len(x) # number of data points
    features = len(x.T) # number of features

    if der == 0: # return cost
        return 1 / (2 * m) * np.sum((h(O, x) - y) ** 2) + np.sum(l * O ** 2)

    if der == 1: # return derivatives
        O_der = np.empty(shape=(features, 1))  # empty array to store derivatives
        for i in range(0, len(O)):
           O_der[i] = (1 / m * np.sum((h(O, x) - y) * x[:,i].reshape(m, 1)))
        return(O_der)


# Normal equations method
def NormalEquationsMethod(x, y):
    O = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x), x)), np.transpose(x)), y)
    return(O)


# Main

# y - array of correct answers (dependent variable)
# x - arrays of other variables (independent variables)
# O - weights (rows correspond to each column of x)

# Generate data
DataPoints = 100
Features = 5

x = np.random.randint(1, 100, size = (DataPoints, Features+1))
x[:,0] = 1
x = x[x[:,1].argsort()]
y = (x[:,1] * np.random.uniform(0.2, 0.5, size=(1, DataPoints))).reshape(DataPoints, 1)
O = np.random.uniform(-0.1, 0.1, size=(Features+1, 1))

# Gradient decent
steps = 100
alpha = 1e-6
l = 0.5
train_history = []  # list to save gradient progress

for step in range(steps):
    O = O * (1 - alpha * l / len(y)) - alpha * J(O, x, y, l, der=1)
    train_history.append(J(O, x, y, l, der=0))
    print(step, " :", J(O, x, y, l, der=0))

pl.plot(train_history)

# Normal equations method
O = NormalEquationsMethod(x, y)

# Plot data and hypothesis
pl.plot(y[:,0], 'ro')
pl.plot(h(O, x))

