"""
Useful expressions common to many neural network applications.
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2013, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

import numpy as np
import theano
from theano.printing import Print
from theano import tensor as T


def softmax_numpy(x):
    """
    .. todo::

        WRITEME properly

    x: a matrix
    returns a vector, with rval[i] being the softmax of row i of x
    """
    stable_x = (x.T - x.max(axis=1)).T
    numer = np.exp(stable_x)
    return (numer.T / numer.sum(axis=1)).T

def pseudoinverse_softmax_numpy(x):
    """
    .. todo::

        WRITEME properly

    x: a vector
    returns y, such that softmax(y) = x
    This problem is underdetermined, so we also impose y.mean() = 0
    """
    rval = np.log(x)
    rval -= rval.mean()
    return rval

def sigmoid_numpy(x):
    """
    .. todo::

        WRITEME
    """
    assert not isinstance(x, theano.gof.Variable)
    return 1. / (1. + np.exp(-x))

def inverse_sigmoid_numpy(x):
    """
    .. todo::

        WRITEME
    """
    return np.log(x / (1. - x))

def arg_of_softmax(Y_hat):
    """
    .. todo::

        WRITEME
    """
    assert hasattr(Y_hat, 'owner')
    owner = Y_hat.owner
    assert owner is not None
    op = owner.op
    if isinstance(op, Print):
        assert len(owner.inputs) == 1
        Y_hat, = owner.inputs
        owner = Y_hat.owner
        op = owner.op
    if not isinstance(op, T.nnet.Softmax):
        raise ValueError("Expected Y_hat to be the output of a softmax, "
                "but it appears to be the output of " + str(op) + " of type "
                + str(type(op)))
    z ,= owner.inputs
    assert z.ndim == 2
    return z

def softmax_ratio(numer, denom):
    """
    .. todo::

        WRITEME
    """

    numer_Z = arg_of_softmax(numer)
    denom_Z = arg_of_softmax(denom)
    numer_Z -= numer_Z.max(axis=1).dimshuffle(0, 'x')
    denom_Z -= denom_Z.min(axis=1).dimshuffle(0, 'x')

    new_num = T.exp(numer_Z - denom_Z) * (T.exp(denom_Z).sum(axis=1).dimshuffle(0, 'x'))
    new_den = (T.exp(numer_Z).sum(axis=1).dimshuffle(0, 'x'))

    return new_num / new_den
