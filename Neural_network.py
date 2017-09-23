import numpy as np
from scipy import optimize
import pylab as pl
from tensorflow.examples.tutorials.mnist import input_data # tf is used just for MNIST dataset

#       Input           Weights         1st hidden       Weights           2nd hidden             Output
#       Layer           Layer           Layer            Layer             Layer                  Layer
#       L[0]            O[0]            L[1]             O[1]              L[2]                   L[3]
#
#       D ## 4+1 ##     I ##  3  ##     D ## 3+1 ##      N   ##  3  ##     D ## 3+1 ##            D ##  2  ##
#       a #       #     n #       #     a #       #      e   #       #     a #       #     ...    a #       #
#       t #50     #     p #5      #     t #50     #      u   #4      #     t #50     #     ...    t #50     #
#         #       #       #       #       #       #      r+1 #       #       #       #     ...      #       #
#         ##     ##       ##     ##       ##     ##          ##     ##       ##     ##              ##     ##
#         Inp             Neur            Neur+1             Neur            Neur+1                 Output
#

# List of variables:
# L_Empty   -   Empty dictionary (NN shell) of chosen size. All neurons equal to zeros. Created using LayerPrep().
# L         -   Filled dictionary (filled NN shell) estimated using inputs and weights. Created using ForwardProp().
# O         -   Dictionary where weights are stored. Created using WightsPrep().
# O_grad    -   Dictionary where gradients of weights are stored. Created using G().
# A         -   Dictionary of z values used in backpropagation. Created using ForwardProp().


# Layer preparation
def LayerPrep(x, y, Neurons, HiddenLayers):
    L_empty = {}                                                                            # Creating empty Dict
    L_empty[0] = np.concatenate((np.zeros(shape=(len(x), 1)) + 1, x), axis=(1))             # Input Layer: L0 = X + Bias
    for i in range(1, HiddenLayers+1):                                                      # Hidden Layers: L1, L2, L3 ...
        L_empty[i] = np.zeros(shape=(len(L_empty[i-1]), Neurons+1))                         # Size of a hidden layer: nrow = nrow of previous layer, ncol = Neurons + 1(bias)
    L_empty[len(L_empty)] = np.zeros(shape=(np.shape(y)))                                   # Output Layer: shape = shape of y
    return L_empty


# Weights preparation
def WeightsPrep(x, y, Neurons, HiddenLayers, L_empty, low, high):
    O = {}                                                                                  # Creating empty Dict
    O[0] = np.random.uniform(low, high, size=(len(x.T)+1, Neurons))                         # Input weights: O0 (L0)
    for i in range(1, HiddenLayers):                                                        # Hidden weights: O1, O2, O3 (L1, L2, L3)
        O[i] = np.random.uniform(low, high, size=(Neurons+1, Neurons))                      # Size of Hidden weights: nrow = Neurons + 1(Bias), ncol = Neurons
    O[len(O)] = np.random.uniform(low, high, size = (Neurons+1, len(y.T)))                  # Creating weights for Output (last L): shape nrow = Neurons + 1(Bias), ncol = nr of Y
    O = ReshapeO(O, L_empty, vect=1)                                                        # Vectorizing O's
    return O


# Forward propagation
def ForwardPropagation(O, L_empty):
    O = ReshapeO(O, L_empty, vect=0)                                                        # Reshaping O's into dict
    L = {}                                                                                  # Creating empty dict to store values of neurons for back propagation
    A = {}                                                                                  # Creating a dict to store z products for back propagation
    L[0] = L_empty[0]                                                                       # L[0] is equalto L_empty[0] = x + bias
    Bias = np.zeros(shape=(len(L_empty[0]), 1)) + 1                                         # Creating Bias outside the loop
    for i in range(1, len(L_empty)):                                                        # A loop for each of hidden layer
        L[i] = h(np.dot(L[i-1], O[i-1]))                                                    # Hidden Layer n = previous hidden layer L(n-1) * its weights O(n-1)
        if i < len(L_empty)-1:                                                              # If because there is no need to add bias for the last layer
            L[i] = np.concatenate((Bias, L[i]), axis=(1))                                   # Adding Bias for each layer
        if i < len(L_empty)-1:                                                              # If because there is no need to have z values for last layer
            A[i] = np.concatenate((Bias, np.dot(L[i-1], O[i-1])), axis=(1))                 # Filling z values for back propagation
    return L, A, L[len(L)-1]


# Computing gradients with back propagation
def G(O, *args):
    y = args[0]
    L, A, y_est = ForwardPropagation(O, L_empty)                                            # Retrieving L, A, and y-est from forward propagation
    O = ReshapeO(O, L_empty, vect=0)                                                        # Reshaping O's into dict
    O_grad = {}                                                                             # Creating empty dict to store gradients
    D = {}                                                                                  # Creating empty dict to store D values
    for i in reversed(range(1, len(O)+1)):                                                  # A loop to estimate D values
        if i == len(O):                                                                     # If because the last D value is differerent
            D[i] = y_est - y                                                                # Last D value is = y_est - y
        elif i == len(O)-1:                                                                 # If because using last D value is exceptional
            D[i] = np.dot(D[i + 1], O[i].T) * derr(A[i])                                    # Last - 1 D value is different because no need to get rid of bias in front
        else:                                                                               # The rest of D values are the same
            D[i] = np.dot(D[i+1][:,1:len(D[i+1].T)], O[i].T) * derr(A[i])                   # D[n] values are = D[n+1](without bias) .* O[n] * derr(A[n])
    for i in range(0, len(O)):                                                              # A loop to estimate O_grads in all layers
        if i == len(O)-1:                                                                   # If because in the last layer there is no need to drop bias that does not exist
            O_grad[i] = np.dot(L[i].T, D[i+1])                                              # O_grad[n] = L[n] .* D[n+1]
        else:                                                                               # Else all layers except the last one to drop biases
            O_grad[i] = np.dot(L[i].T, D[i+1][:,1:len(D[i+1].T)])                           # O_grad[n] = L[n] .* (D[n+1] - bias)
    for i in range(0, len(O)):                                                              # Looping through each Layer's O's to separately est lamda grads
        O_grad_l = np.concatenate((np.zeros((1,len(O[i].T))), O[i][1:]), axis=(0))          # All the O's except the bias row, which are 0's
        O_grad[i] = (1/len(y)) * (O_grad[i] + l * O_grad_l)                                 # O_grad = 1/m * (O_grad + O_grad_l)
    O_grad = ReshapeO(O_grad, L_empty, vect=1)                                              # Vectorizing O_grad
    return O_grad


# Activation function
def h(x):
    return 1 / (1+np.exp(1) ** -x)


# Derivative of activation function
def derr(x):
    return h(x) * (1 - h(x))


# Cost function
def J(O, *args):
    L_empty = args[2]
    y_est = ForwardPropagation(O, L_empty)[2]
    y = args[0]
    l = args[1]
    O = ReshapeO(O, L_empty, vect=0)
    lambda_cost = 0                                                                         # Lambda cost is estimated separately
    for i in range(len(O)):                                                                 # Looping through all Layer's O's
        lambda_cost = lambda_cost + (l / (2 * len(y))) * np.sum(O[i][1:len(O[i]):] ** 2)    # Estimating Lambda cost, not including the top row O's because it belongs to bias
    total_cost = (1/len(y)) * np.sum(-y * np.log(y_est) - (np.ones(np.shape(y)) - y) * np.log(1 - y_est)) + lambda_cost
    return total_cost


# Rolling and unrolling O's
def ReshapeO(O, L_empty, vect):
    O_reshaped = 'Error'                                                                    # If wrong vect parameter is used
    if vect == 1:                                                                           # If because for vect == 1, normal O is transformed to a vector
        O_reshaped = np.ndarray.flatten(np.zeros((0,0)))                                    # Empty vector to concetanet values to
        for i in range(0, len(O)):                                                          # A loop for O's in each Layer
            O_current = np.ndarray.flatten(O[i])                                            # Taking O's and reshaping into vector
            O_reshaped = np.concatenate((O_reshaped, O_current), axis=(0))                  # Concetanating previous vectorized values of O's with current vectorized O
    if vect == 0:                                                                           # If because for vect == 0, vectorized O's are being put back to its original form
        O_reshaped = {}                                                                     # Creating empty dict to store the values of reshaped O
        start = 0                                                                           # Counter used to calculate the starting point in vectorized O's
        for i in range(0, len(L_empty)-1):
            if i == len(L_empty)-2:
                nrow = len(L_empty[i].T)
                ncol = len(L_empty[i+1].T)
            else:
                nrow = len(L_empty[i].T)
                ncol = len(L_empty[i+1].T)-1
            O_reshaped[i] = O[start:start+(ncol*nrow)].reshape((nrow,ncol))                 # Taking appropriate values from vectorized form, reshaping it and storing in O_reshaped
            start = start + ncol * nrow                                                     # Updating starting point
    return O_reshaped


# Numerical gradient checking
def checkgradients(O, alpha, show, biggestdiff):
    numgrad = np.array(O) * 0
    O_high = np.array(O)
    O_low = np.array(O)
    for i in range(0, len(O)):
        O_high[i] = O_high[i] + alpha
        O_low[i] = O_low[i] - alpha
        numgrad[i] = (J(O_high, *args) - J(O_low, *args)) / (2*alpha)
        print(i, 'of', len(O))
    fungrad = np.reshape(G(O, *args), (len(G(O, *args)), 1))
    numgrad = np.reshape(numgrad, (len(numgrad),1))
    if show == 1:
        print('numgrad and fungrad \r', np.concatenate((numgrad, fungrad), axis=(1)))
    if biggestdiff == 1:
        differences = np.abs(numgrad - fungrad)
        differences = np.around(np.sort(differences, axis=0, kind='quicksort'), decimals = 2)
        print('5 biggest differences', differences[(len(differences)-5):len(differences),:])
    return print('Sum of differences', np.sum(np.abs(fungrad - numgrad)))


# Estimating accuracy
def accuracy(y, y_est, threshold=0.5):
    y_est_threshold = (y_est > threshold) * 1
    true_positives = np.sum(np.logical_and(y_est_threshold == 1, y == 1))               # Of 10 predicted 1's, 8 are actually 1's
    if np.count_nonzero(y_est_threshold == 1) == 0:
        precision = np.nan
    else:
        precision = true_positives / np.count_nonzero(y_est_threshold == 1)              # 8 that were true / by all that were predicted to be 1's
    recall = true_positives / (len(y) * len(y.T))
    if (precision + recall) == 0:
        F_1 = np.nan
    else:
        F_1 = 2 * (precision * recall) / (precision + recall)
    return round(float(precision)*100, 1), round(float(recall)*100, 1), round(float(F_1)*100, 1), \
           np.divide(np.sum([y == ([y_est > threshold])]), (y_est.shape[0] * y_est.shape[1])) * 100


# Plotting
def plotresults(O_result):
    for i in range(1, len(O_result[1])):
        Iteration = i
        O_selected = O_result[1][i]
        pl.figure(1)

        pl.subplot(221)
        pl.plot(Iteration, np.around(J(O_selected, *args), 4), 'bo',
                Iteration, np.around(J(O_selected, *args_cv), 4), 'ro', markersize=4)
        pl.ylabel('Error')

        pl.subplot(222)
        pl.plot(Iteration, accuracy(y, ForwardPropagation(O_selected, L_empty)[2], 0.5)[0], 'bo',
                Iteration, accuracy(y_cv, ForwardPropagation(O_selected, L_empty_cv)[2], 0.5)[0], 'ro', markersize=4)
        pl.ylabel('Precision')

        pl.subplot(223)
        pl.plot(Iteration, np.average(np.abs(O_selected)), 'go',
                Iteration, np.max(np.abs(O_selected)), 'go',
                Iteration, np.min(np.abs(O_selected)), 'go', markersize=4)
        pl.ylabel('Weights (min, max, avg)')

        pl.subplot(224)
        pl.plot(Iteration, 1)
        pl.ylabel('Nothing')


# Optimization callback function
def interm(O_interm):
    print('Train error %.2f:' % J(O_interm, *args),
          'Precision, Recall, F1, Accuracy: %.2f, %.2f, %.2f, %.2f' % accuracy(args[0], ForwardPropagation(O_interm, L_empty)[2], 0.5), '  |  ',
          'Test error: %.2f' % J(O_interm, *args_cv),
          'Precision, Recall, F1, Accuracy: %.2f, %.2f, %.2f, %.2f' % accuracy(args_cv[0], ForwardPropagation(O_interm, L_empty_cv)[2], 0.5))


def initializeNN(Neurons, HiddenLayers, l, O_min, O_max, x, y, x_cv, y_cv):

    # Initializing Training Neural Net
    L_empty = LayerPrep(x, y, Neurons, HiddenLayers)
    O = WeightsPrep(x, y, Neurons, HiddenLayers, L_empty, O_min, O_max)
    L, A, y_est = ForwardPropagation(O, L_empty)
    args = (y, l, L_empty)

    # Initializing cross-validation Neural Net
    L_empty_cv = LayerPrep(x_cv, y_cv, Neurons, HiddenLayers)
    L_cv, A_cv, y_est_cv = ForwardPropagation(O, L_empty_cv)
    args_cv = (y_cv, l, L_empty_cv)

    print('Train error:', np.around(J(O, *args), 2), 'Precision:',
          accuracy(y, ForwardPropagation(O, L_empty)[2], 0.5)[0], '  |  ''Test error:',
          np.around(J(O, *args_cv), 2), 'Precision:',
          accuracy(y_cv, ForwardPropagation(O, L_empty_cv)[2], 0.5)[0])

    return O, L_empty, L, A, y_est, args, L_empty_cv, L_cv, A_cv, y_est_cv, args_cv, l

# Main

# Data (MNIST dataset from tensorflow). Data points - rows; features - columns
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x_full = mnist.train.images
y_full = mnist.train.labels
x = x_full[0:10000,]
y = y_full[0:10000,]
x_cv = x_full[10000:13000,]
y_cv = y_full[10000:13000,]

# Initialize Neural Network
O, L_empty, L, A, y_est, args, L_empty_cv, L_cv, A_cv, y_est_cv, args_cv, l = initializeNN(
                                                                                        Neurons=5,
                                                                                        HiddenLayers=3,
                                                                                        l=0.5,
                                                                                        O_min=-0.1,
                                                                                        O_max=0.1,
                                                                                        x = x,
                                                                                        y = y,
                                                                                        x_cv = x_cv,
                                                                                        y_cv = y_cv)

# Optimization
O_result = optimize.fmin_cg(J, O, fprime=G, args=args, maxiter=150, disp=True, callback=interm, retall=True)

# Plot results
plotresults(O_result)

# Save opt O's to continue optimise further
O = O_result[0]

#Checking gradients
checkgradients(O, alpha=0.0001, show=1, biggestdiff=0) # do for small nets only