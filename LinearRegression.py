import numpy as np
import matplotlib.pyplot as plt
import copy, math


def predict(X, w, b):
    """
    calculate prediction based on weights w, bias b, and features X
    """
    return np.dot(X, w) + b

def compute_cost(X, y, w, b, lambda_=0):
    """
    calculates the cost of a predictive model with weights w and bias b
    cost function used: (first image) https://math.stackexchange.com/questions/2202545/why-using-squared-distances-in-the-cost-function-linear-regression
    """

    m,n = X.shape # get the # of data points
    totalCost = 0

    for i in range (m):
        f_wb_i = np.dot(X[i], w) + b
        totalCost += (f_wb_i - y[i])**2
    totalCost /= 2*m

    # extra cost for regularization, does nothing is lambda is 0 (no regulariation)
    regularizationCost = 0;
    for i in range(n):
        regularizationCost += (w[i])**2
    regularizationCost *= (lambda_/(2*m));
    
    totalCost += regularizationCost
    return totalCost

def compute_gradient(X, y, w, b, lambda_ = 0):
    """
    computes and returns the gradient for one step of gradient descent
    gradient function used: https://stackoverflow.com/questions/33847827/gradient-descent-for-more-than-2-theta-values 
    """

    m, n = X.shape          # m = # of examples, n = # of features in each example
    dj_dw = np.zeros((n,))  # gradient/partial derivative of cost function in regards to w
    dj_db = 0               # gradient/partial derivative of cost function in regards to b

    for i in range(m):
        error = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] += error * X[i][j]
        dj_db += error
    dj_dw /= m
    dj_db /= m

    # regularization, if lambda default (0) then this has no effect
    for i in range(n):
        dj_dw[i] += (lambda_/m) * w[i]

    return dj_dw, dj_db

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_=0):
    """
    performs batch gradient descent: performs num_iters # of gradient measurements and adjusts
    weights w and bias b accordingly with learning rate alpha

    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter
    """

    cost_history = []
    w = copy.deepcopy(w_in) # deep copy to avoid changing the input 
    b = b_in
    
    for i in range(num_iters):
        # if(num_iters % 10 == 0):
            # print(w)
        dj_dw, dj_db = gradient_function(X, y, w, b, lambda_)
        w -= alpha * dj_dw
        b -= alpha * dj_db

        if i < 100000:
            cost_history.append(cost_function(X, y, w, b, lambda_))

        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {cost_history[-1]:8.2f}   ")

    return w, b, cost_history

def zscore_normalize_features(X):
    """
    normalizes the features X through zscore normalization by column
    """

    means = np.mean(X, axis=0)  # find the mean of each column/feature
    sigma = np.std(X, axis=0)   # standard deviation of each col/feature
    X_norm = (X - means)/sigma

    return X_norm, means, sigma

"""
Data Parsing/cleaning 
"""
data = np.loadtxt("./Data/petrol_usage.txt", skiprows=41) # load data
data = np.delete(data, 0, 1)    # delete first col (index)
labels = ['petrol tax', 'per capita income', 'miles of paved driveway', 'proportion of drivers']

# break data into features and outcomes
X = np.delete(data, 4, 1)
X, means, sigma = zscore_normalize_features(X)
# np.s_[0:4] means 0 up to 4 cols get deleted, the 1 = col, 0 = row
Y = np.delete(data, np.s_[0:4], 1)
# flatten to make it go from a 2d array with a bunch of 1 entry arrays
# to a 1d array with the same # of entries 
Y = Y.flatten()
# print(f"X Shape: {X.shape}, X Type:{type(X)})")
# print(f"y Shape: {Y.shape}, y Type:{type(Y)})")
# print(X)
# print(Y)

"""
Data Plotting/Visualization
"""
# fig, ax = plt.subplots(1,4,figsize=(12,5),sharey=True)
# for i in range(len(ax)):
#     # [: x,y] syntax = row , column
#     ax[i].scatter(X[:,i], Y, c="blue")
#     ax[i].set_xlabel(labels[i])
# ax[0].set_ylabel("petrol used")
# plt.show()

"""
test data creation
"""
# X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
# y_train = np.array([460, 232, 178])
# print(f"X Shape: {X_train.shape}, X Type:{type(X_train)})")
# print(X_train)
# print(f"y Shape: {y_train.shape}, y Type:{type(y_train)})")
# print(y_train)
# b_init = 785.1811367994083
# w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
# print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")


"""
Thoughts after visualising dataset:
    There doesn't seem to be any obvious trend in the data (at least linearly)
    except for in the "proportion of drivers" vs petrol usage graph. I predict
    that the weight for the proportion of drivers feature will be larger relative
    to the other features.
"""

# """ predict() test """
# # get a row from our training data
# x_vec = X_train[0,:]
# print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")
# # make a prediction
# f_wb = predict(x_vec,w_init, b_init)
# print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")

# """ compute_cost() test """
# cost = compute_cost(X_train, y_train, w_init, b_init)
# print(f'Cost at optimal w : {cost}')

# """ compute_Gradient test """
# tmp_dj_dw, tmp_dj_db= compute_gradient(X_train, y_train, w_init, b_init)
# print(f'dj_db at initial w,b: {tmp_dj_db}')
# print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')

# """ gradient descent testing """
# # initialize parameters
# initial_w = np.zeros_like(w_init)
# initial_b = 0.
# # some gradient descent settings
# iterations = 1000
# alpha = 5.0e-7
# # run gradient descent 
# w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations)
# print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
# m,_ = X_train.shape
# for i in range(m):
#     print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")

""" Regression on data """
input_w = np.zeros(X.shape[1])
input_b = 0
iter = 10000
alpha = 5.0e-2
# initial alpyha of 5.0e-7 was too large and leading to infinite cost, 
w_predict, b_predict, history = gradient_descent(X, Y, input_w, input_b, compute_cost, compute_gradient, alpha, iter, lambda_=3)

print(w_predict)
for i in range(X.shape[0]):
    print(f"prediction: {predict(X[i], w_predict, b_predict):0.2f}, target value: {Y[i]}")


"""
Notes after implementing regression without feature scaling or regularization:
    At first I ran into an overflow problem which I figured was due to a large learning rate
    and after lowering the learning rate by 100x, the cost successfully went down each iteration.

    The results of the regression were a little underwhelming, with the cost reaching its lowest 
    at around 10900, and each prediction being around 100 off of the real value. Also, the weights
    of the features that were produced through gradient descent were not what I predicted, and seem
    to emphasize the 2nd input rather than the 4th like I predicted. Hopefully implementing feature
    scaling and regularization will fix this.
    weights = [0.00342409 0.12526768 0.00576827 0.00058011]
    cost = 10900
"""

"""
Notes on implementing feature scaling after regular linear regression:
    After successfully normalizing each feature, no longer got valid results. Fixed by increasing learning
    rate/alpha by 10^7, resulting in better predictions and weights that are closer to prediction
    weights = [-32.7310515  -37.79691223  -8.3813148   73.3569154 ]
    cost = 1969

    while there is probably room for improvement through regularization, the weights now heavily emphasize
    the 4th feature (proportion of drivers) like initial prediction
"""

"""
Notes on implementing regularization:
    Overall, no obvious improvements from implementing regularization. as lambda went up the cost went up, which
    is to be expected because its usage is to generalize the algorithm.
    weights (lambda = 1) = [-32.41868846 -36.79987493  -8.19691574  71.80614686]
    cost = 2050
    weights (lambda = 3) = [-31.76500298 -34.95554814  -7.81564339  68.91779409]
    cost = 2203
"""

