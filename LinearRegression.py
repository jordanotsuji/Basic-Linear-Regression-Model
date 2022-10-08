import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("./Data/petrol_usage.txt", skiprows=41) # load data
data = np.delete(data, 0, 1)    # delete first col (index)
labels = ['petrol tax', 'per capita income', 'miles of paved driveway', 'proportion of drivers']

# break data into features and outcomes
X = np.delete(data, 4, 1)
# np.s_[0:4] means 0 up to 4 cols get deleted, the 1 = col, 0 = row
Y = np.delete(data, np.s_[0:4], 1)
# flatten to make it go from a 2d array with a bunch of 1 entry arrays
# to a 1d array with the same # of entries 
Y = Y.flatten()

fig, ax = plt.subplots(1,4,figsize=(12,5),sharey=True)
for i in range(len(ax)):
    # [: x,y] syntax = row , column
    ax[i].scatter(X[:,i], Y)
    ax[i].set_xlabel(labels[i])
ax[0].set_ylabel("petrol used")
plt.show()

"""
Thoughts after visualising dataset:
    There doesn't seem to be any obvious trend in the data (at least linearly)
    except for in the "proportion of drivers" vs petrol usage graph. I predict
    that the weight for the proportion of drivers feature will be larger relative
    to the other features.
"""

def predict(X, w, b):
    """
    calculate prediction based on weights w, bias b, and features X
    """
    return np.dot(X, w) + b

def compute_cost(X, y, w, b):
    """
    calculate the cost of a predictive model with weights w and bias b
    cost function used: (first image) https://math.stackexchange.com/questions/2202545/why-using-squared-distances-in-the-cost-function-linear-regression
    """

    m = X.shape[0] # get the # of data points
    totalCost = 0

    for i in range (m):
        f_wb_i = np.dot(X[i], w) + b
        totalCost += (f_wb_i - y[i])**2
    totalCost /= 2*m
    return totalCost


