import numpy as np

def inverse_sigmoid_numpy(x):
    return np.log( - x / ( x - 1.))
