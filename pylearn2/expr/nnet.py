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

def sigmoid_numpy(x):
    assert not isinstance(x, theano.gof.Variable)
    return 1. / (1. + np.exp(-x))

def inverse_sigmoid_numpy(x):
    return np.log(x / (1. - x))

def arg_of_softmax(Y_hat):
    assert hasattr(Y_hat, 'owner')
    owner = Y_hat.owner
    assert owner is not None
    op = owner.op
    if isinstance(op, Print):
        assert len(owner.inputs) == 1
        Y_hat, = owner.inputs
        owner = Y_hat.owner
        op = owner.op
    assert isinstance(op, T.nnet.Softmax)
    z ,= owner.inputs
    assert z.ndim == 2
    return z

def softmax_ratio(numer, denom):

    numer_Z = arg_of_softmax(numer)
    denom_Z = arg_of_softmax(denom)
    numer_Z -= numer_Z.max(axis=1).dimshuffle(0, 'x')
    denom_Z -= denom_Z.min(axis=1).dimshuffle(0, 'x')

    new_num = T.exp(numer_Z - denom_Z) * (T.exp(denom_Z).sum(axis=1).dimshuffle(0, 'x'))
    new_den = (T.exp(numer_Z).sum(axis=1).dimshuffle(0, 'x'))

    return new_num / new_den
