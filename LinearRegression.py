import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("./Data/petrol_usage.txt", skiprows=41) # load data
data = np.delete(data, 0, 1)    # delete first col (index)

# break data into features and outcomes
X = np.delete(data, 4, 1)
# np.s_[0:4] means 0 up to 4 cols get deleted, the 1 = col, 0 = row
Y = np.delete(data, np.s_[0:4], 1)
# flatten to make it go from a 2d array with a bunch of 1 entry arrays
# to a 1d array with the same # of entries 
Y = Y.flatten()

print(X[:,1])
print(Y)
# plt.scatter(X[:,0], Y, c="blue")
plt.scatter(X[:,1], Y, c="red")
plt.ylabel("petrol used")
plt.xlabel("average income (dollars)")
plt.show()
