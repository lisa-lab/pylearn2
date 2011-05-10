"""Logistic regression.

Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability. 

Mathematically, this can be written as:

.. math::
  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
                &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}


The output of the model or prediction is then done by taking the argmax of 
the vector whose i'th element is P(Y=i|x).

.. math::

  y_{pred} = argmax_i P(Y=i|x,W,b)
"""

__docformat__ = 'restructedtext en'

# Standard library imports
import cPickle as pickle
import gzip, os, sys, time

# Third-party imports
import numpy
import theano
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# Local imports
from .base import Block
from .utils import sharedX

floatX = theano.config.floatX

class LogisticRegressionLayer(Block):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W` 
    and bias vector :math:`b`. Classification is done by projecting data 
    points onto a set of hyperplanes, the distance to which is used to 
    determine a class membership probability.

    This class contains only the part that computes the output (prediction),
    not the classification cost, see cost.OneHotCrossEntropy for that.
    """

    def __init__(self, nvis, nclasses):
        """Initialize the parameters of the logistic regression

        Parameters
        ----------
        nvis : int
            number of input units, the dimension of the space in which
            the datapoints lie.

        nclasses : int
            number of output units, the dimension of the space in which
            the labels lie.
        """

        assert nvis >= 0, "Number of visible units must be non-negative"
        assert nclasses >= 0, "Number of classes must be non-negative"

        self.nvis = nvis
        self.nclasses = nclasses

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out) 
        self.W = sharedX(numpy.zeros((n_in,n_out)), name='W', borrow=True)
        # initialize the baises b as a vector of n_out 0s
        self.b = sharedXtheano.shared((n_out,), name='b', borrow=True)

        # parameters of the model
        self._params = [self.W, self.b]


    def p_y_given_x(self, input):
        # compute vector of class-membership probabilities in symbolic form
        return tensor.nnet.softmax(tensor.dot(input, self.W)+self.b)

    def predict_y(self, input):
        # compute prediction as class whose probability is maximal in 
        # symbolic form
        return tensor.argmax(self.p_y_given_x(input), axis=1)

    def __call__(self, input):
        return self.p_y_given_x(input)

