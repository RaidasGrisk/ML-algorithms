import numpy as np
import math
import matplotlib
import pylab as pl

### Hypothesis
def h(O, x):
    return 1 / (1 + math.exp(1) ** (-np.dot(x, O)))

### Cost function
def J(O, x, y, l, derr):
    if derr == 0:
        #return (-1/len(x)) * np.sum(y * np.log(h(O)) + (1-y) * np.log(1 - h(O))) + np.sum(l * O[-1] ** 2) # logistic cost function
        return 1 / 2 * np.shape(x)[0] * np.sum((h(O, x) - y) ** 2) + np.sum(l * O ** 2) # Usual cost function
    if derr == 1:
        O_derr = np.empty(shape=(np.shape(x)[1], 1))  # an ampty list to be filled if solving for derrivatives
        for i in range(0, len(O)):
            O_derr[i] = np.sum((h(O, x) - y) * x[:, i].reshape(np.shape(x)[0], 1))
        return(O_derr)

### Gardient decent
def GradientDecent(steps, alpha, O, x, y, printSteps = "False", plotSteps = "False"):
    StepsData = np.empty(shape = (steps, 2))
    for i in range(1, steps+1):
        O_new = O * (1 - alpha*l/len(y)) - alpha * J(O, x, y, l, derr=1)
        O = O_new
        StepsData[i - 1, 0] = i
        StepsData[i - 1, 1] = J(O, x, y, l, derr=0)
        if printSteps == "True":
            print(i, " :", J(O, x, y, l, derr = 0))
    if plotSteps == "True":
        pl.plot(StepsData[:, 0], StepsData[:, 1])
    return(O)

#################
### Dashboard ###
#################

# y - array of correct answers
# x - matrix of arrays of other variables
# O - an array of initial guess of weights. Each row, correspond to each column of x

### Data
DataPoints = 100
NrofO = 2
x = np.sort(np.random.randint(0, 100, size = (NrofO+1, DataPoints)))
x = np.transpose(x)
y = np.sort(np.random.randint(0, 2, size = (1, DataPoints)))
y = y.reshape(DataPoints, 1)
x[:,0] = 1

### Gradient decent
steps = 10000
alpha = 0.0001
l = 0.8
O = np.random.uniform(-0.9, 0.9, size=(np.shape(x)[1], 1))

O = GradientDecent(steps, alpha, O, x, y, printSteps = "True", plotSteps = "True")

### Data and hypothesis
pl.plot(np.arange(1, np.shape(x)[0]+1), y[:,0], 'ro')
pl.plot(np.arange(1, np.shape(x)[0]+1), np.concatenate(h(O, x)))

pl.plot(np.arange(1, np.shape(x)[0]+1), x[:,1])

### normal equations method
def NormalEquationsMethod(x, y):
    O = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x), x)), np.transpose(x)), y)
    return(O)
