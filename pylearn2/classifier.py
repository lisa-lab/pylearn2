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

# Third-party imports
import numpy
from theano import tensor

# Local imports
from pylearn2.base import Block
from pylearn2.utils import sharedX
from pylearn2.space import VectorSpace
from pylearn2.models import Model

class LogisticRegressionLayer(Block, Model):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.

    This class contains only the part that computes the output (prediction),
    not the classification cost, see cost.OneHotCrossEntropy for that.
    """
    def __init__(self, nvis, nclasses):
        """Initialize the parameters of the logistic regression instance.

        Parameters
        ----------
        nvis : int
            number of input units, the dimension of the space in which
            the datapoints lie.

        nclasses : int
            number of output units, the dimension of the space in which
            the labels lie.
        """

        super(LogisticRegressionLayer, self).__init__()

        assert nvis >= 0, "Number of visible units must be non-negative"
        self.input_space = VectorSpace(nvis)
        self.output_space = VectorSpace(nclasses)
        assert nclasses >= 0, "Number of classes must be non-negative"

        self.nvis = nvis
        self.nclasses = nclasses

        # initialize with 0 the weights W as a matrix of shape (nvis, nclasses)
        self.W = sharedX(numpy.zeros((nvis, nclasses)), name='W', borrow=True)
        # initialize the biases b as a vector of nclasses 0s
        self.b = sharedX(numpy.zeros((nclasses,)), name='b', borrow=True)

        # parameters of the model
        self._params = [self.W, self.b]

    def p_y_given_x(self, inp):
        """TODO: docstring"""
        # compute vector of class-membership probabilities in symbolic form
        return tensor.nnet.softmax(tensor.dot(inp, self.W) + self.b)

    def predict_y(self, inp):
        """TODO: docstring"""
        # compute prediction as class whose probability is maximal in
        # symbolic form
        return tensor.argmax(self.p_y_given_x(inp), axis=1)

    def __call__(self, inp):
        """TODO: docstring"""
        return self.p_y_given_x(inp)
