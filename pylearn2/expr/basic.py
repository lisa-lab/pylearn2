import numpy as np
import theano.tensor as T
from pylearn2.utils import as_floatX

def numpy_norms(W):
    """ returns a vector containing the L2 norm of each
column of W, where W and the return value are
numpy ndarrays """
    return np.sqrt(1e-8+np.square(W).sum(axis=0))

def theano_norms(W):
    """ returns a vector containing the L2 norm of each
column of W, where W and the return value are symbolic
theano variables """
    return T.sqrt(as_floatX(1e-8)+T.sqr(W).sum(axis=0))

def full_min(var):
    """ returns a symbolic expression for the value of the minimal
    element of symbolic tensor. T.min does something else as of
    the time of this writing. """
    return var.min(axis=range(0,len(var.type.broadcastable)))

def full_max(var):
    """ returns a symbolic expression for the value of the maximal
        element of a symbolic tensor. T.max does something else as of the
        time of this writing. """
    return var.max(axis=range(0,len(var.type.broadcastable)))


def multiple_switch(*args):
    """
    Applies a cascade of ifelse. The output will be a Theano expression
    which evaluates:
        if args0:
            then arg1
        elif arg2:
            then arg3
        elif arg4:
            then arg5
        ....
    """
    if len(args) == 3:
        return T.switch(*args)
    else:
        return T.switch(args[0],
                         args[1],
                         multiple_switch(*args[2:]))

