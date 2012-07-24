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
import theano
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

class CumulativeProbabilitiesLayer(LogisticRegressionLayer):
    """
    A layer whose output is seen as a discrete cumulative distribution
    function, i.e. unit i outputs p(y <= i | x).

    To ensure that the outputs are in ascending order, the weights
    matrix is shared between all units and the bias b_i are transformed
    into
            c_i = c_{i-1} + softplus(b_i), c_0 = b_0,
    so that they are in ascending order.

    The outputs are used to compute p_y_given_x as
            p(y = i | x) = p(y <= i | x) - p(y <= i - 1 | x)

                             |  output_i - output_{i-1}, i > 0
                         =  -+
                             |  output_i               , i = 0

    In the special case where some units are saturated, some
    p(y = i | x) = 0 may appear, which can cause problem with regular
    negative log likelihood. It is recommended that the cost function used
    for this layer relies instead on p(y <= i | x), such as
    cost = -sum_over_i(indicator(y <= i) * log(p(y <= i | x)) +
                       indicator(y > i) * log(1 - p(y <= i | x)))
    """

    def __init__(self, nvis, nclasses):
        """
        Initialize the parameters, with W being shared between all units.

        :type nvis: int
        :param nvis: number of input units, the dimension of the space in
                     which the datapoints lie.

        :type nclasses: int
        :param nclasses: number of output units, the dimension of the space
                         in which the labels lie.
        """
        super(CumulativeProbabilitiesLayer, self).__init__(nvis, nclasses)

        self.W = sharedX(numpy.zeros((nvis, 1)), name='W', borrow=True)
        self._params = [self.W, self.b]


    def p_y_ie_n(self, inp):
        """
        Computes the p(y <= i | x) vector given an input.

        The implementation of this function relies on transformation
        matrices, which are explained within the code.

        :type inp: theano.tensor.TensorType
        :param inp: the input used to compute p_y_ie_n
        """
        # As explained in the class docstring, to ensure that the outputs
        # are in ascending order, the W weights matrix is shared between
        # units and the bias vector is transformed so its elements are in
        # ascending order. We use
        #
        #  c_0 = b_0
        #  c_i = c_{i-1} + softplus(b_i)
        #      = c_{i-2} + softplus(b_{i-1}) + softplus(b_i)
        #      = ... = c_0 + softplus(b_1) + ... + softplus(b_i)
        #
        # which can be matricially represented as
        #
        # | c_0 |   | 1 0 ... 0 |   | b_0 |   | 0 0 0 ... 0 |           | b_0 |
        # | c_1 | = | 1 0 ... 0 | * | b_1 | + | 0 1 0 ... 0 | * softplus| b_1 |
        # | ... |   |     ...   |   | ... |   |     ...     |           | ... |
        # | c_n |   | 1 0 ... 0 |   | b_n |   | 0 1 1 ... 1 |           | b_n |
        #
        # or
        #
        # C = K_1 * B + K_2 * softplus(B) 
        #
        # Here we generate K_1. Since B is a 1-dimension vector, we must
        # transpose K_1 to make it work.
        k1_val = numpy.zeros((self.nclasses, self.nclasses),
                             dtype=theano.config.floatX)
        k1_val[:, 0] = 1.0
        k1_val = numpy.transpose(k1_val)
        k1 = theano.shared(value=k1_val, name='k1')
        # Here we generate K_2, which is transposed for the same reason.
        # We first create a (nclasses - 1) x (nclasses - 1) lower triangular
        # matrix and then append the necessary left and upper zeros.
        k2_val = numpy.ones((self.nclasses - 1, self.nclasses - 1),
                            dtype=theano.config.floatX)
        for i in xrange(self.nclasses - 1):
            for j in xrange(self.nclasses - 1):
                if(j > i):
                    k2_val[i, j] = 0
        k2_val = numpy.append(numpy.zeros((self.nclasses - 1, 1),
                                          dtype=theano.config.floatX),
                              k2_val, axis=1)
        k2_val = numpy.append(numpy.zeros((1, self.nclasses),
                                          dtype=theano.config.floatX),
                              k2_val, axis=0)
        k2_val = numpy.transpose(k2_val)
        k2 = theano.shared(value=k2_val, name='k2')

        # The K_1 * B term
        b0_term = tensor.dot(self.b, k1)

        # The K_2 * softplus(B) term
        softplus_term = tensor.dot(tensor.nnet.softplus(self.b), k2)

        # The constructed bias that is ensured to be in ascending order
        c = b0_term + softplus_term

        # Since W is a columns vector, we need to expand it into a matrix.
        weights_constructor_val = numpy.ones((1, self.nclasses))
        weights_constructor = theano.shared(value=weights_constructor_val,
                                            name='weights_constructor')
        weights_matrix = tensor.dot(self.W,  weights_constructor)

        # Compute p(y <= i | x)
        return tensor.nnet.sigmoid(tensor.dot(inp, weights_matrix) + c)

    def p_y_given_x(self, inp):
        """
        Computes p(y = i | x) as p(y <= i | x) - p(y <= i - 1 | x).

        As mentionned in the p_y_ie_n function, the cost function should
        not rely on p_y_given_x because of the zero probabilities that can
        arise if the output units are saturated. It is instead recommended
        to use p_y_ie_n in the cost function expression.

        :type inp: theano.tensor.TensorType
        :param inp: the input used to compute p_y_given_x
        """
        # The expression for p(y = i | x) can be represented in matricial
        # form as
        #
        # P = A * P_prime, P = p(y = i | x), P_prime = p(y <= i | x),
        #                  A = |  1  0  0  0  ...  0  0  |
        #                      | -1  1  0  0  ...  0  0  |
        #                      |  0 -1  1  0  ...  0  0  |
        #                      |          ...      0  0  |
        #                      |  0  0  0  0  ... -1  1  |
        #
        # Here we compute A_transposed because p_y_given_x is a 1-dimension
        # vector.
        a = numpy.eye(self.nclasses, dtype=theano.config.floatX)
        for i in xrange(self.nclasses):
            if(i > 0):
                a[i - 1, i] = -1

        p_y_given_x_mat = theano.shared(value=a, name='p_y_given_x_mat')

        # Compute p_y_given_x
        return tensor.dot(self.p_y_ie_n(inp), p_y_given_x_mat)
