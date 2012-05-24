import numpy as np

def sigmoid_numpy(x):
    return 1. / (1. + np.exp(-x))

def inverse_sigmoid_numpy(x):
    return np.log( - x / ( x - 1.))
