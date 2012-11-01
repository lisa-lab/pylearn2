import numpy as np
import theano

def sigmoid_numpy(x):
    assert not isinstance(x, theano.gof.Variable)
    return 1. / (1. + np.exp(-x))

def inverse_sigmoid_numpy(x):
    return np.log(x / (1. - x))
