import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("./Data/petrol_usage.txt", skiprows=41) # load data
data = np.delete(data, 0, 1)    # delete first col (index)

# break data into features and outcomes
X = np.delete(data, 4, 1)
# np.s_[0:4] means 0 up to 4 cols get deleted, the 1 = col, 0 = row
Y = np.delete(data, np.s_[0:4], 1)

