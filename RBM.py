import numpy as np

def binarize_data(input):
    threshold, upper, lower = 0.5, 1, 0
    np.where(input>=threshold, upper, lower)

def load_data(file):
    data = np.loadtxt(file, dtype='float', delimiter=',')
    X = data[:, :-1]
    binarize_data(X)
    return X

X_train = load_data(load_data("digitstrain.txt"))
print X_train