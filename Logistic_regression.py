import numpy as np
import pylab as pl


# Hypothesis
def h(O, x):
    return 1 / (1 + np.exp(1) ** (-np.dot(x, O)))


# Cost function
def J(O, x, y, l, der):

    m = len(x) # number of data points
    features = len(x.T) # number of features

    if der == 0: # return cost
        return (-1/m) * np.sum(y * np.log(h(O, x)) + (1-y) * np.log(1 - h(O, x))) + np.sum(l * O[-1] ** 2)

    if der == 1: # return derivative
        O_der = np.empty(shape=(features, 1)) # empty array to store derivatives
        for i in range(0, len(O)):
            O_der[i] = np.sum((h(O, x) - y) * x[:,i].reshape(m, 1))
        return(O_der)


# Main

# y - array of correct answers (dependent variable)
# x - arrays of other variables (independent variables)
# O - weights (rows correspond to each column of x)

### Data
DataPoints = 100
Features = 5

x = np.sort(np.random.randint(-2, 2, size=(DataPoints, Features+1)), axis=0)
x[:,0] = 1
y = np.sort(np.random.randint(0, 2, size = (DataPoints, 1)), axis=0)
O = np.random.uniform(-0.1, 0.1, size=(Features+1, 1))

# Gradient decent
steps = 1000
alpha = 1e-5
l = 0.2
train_history = []  # list to save gradient progress

for step in range(steps):
    O = O * (1 - alpha * l / len(y)) - alpha * J(O, x, y, l, der=1)
    train_history.append(J(O, x, y, l, der=0))
    print(step, " :", J(O, x, y, l, der=0))

pl.plot(train_history)

### Data and hypothesis
pl.plot(y, 'ro')
pl.plot(h(O, x))

