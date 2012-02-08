import numpy as np
import theano.tensor as T
from theano.tensor import TensorType
from theano import config

class Space:
    """ Defines a vector space that can be transformed by a linear operator """

    def get_origin(self):
        """ Returns the origin in this space """
        raise NotImplementedError()

    def get_origin_batch(self, n):
        """ Returns a batch of n copies of the origin """
        raise NotImplementedError()

    def make_theano_batch(self, name = None, dtype = None):
        """ Returns a theano tensor capable of representing a batch of points
            in the space """

        raise NotImplementedError()


class VectorSpace(Space):
    """ Defines a space whose points are defined as fixed-length vectors """

    def __init__(self, dim):
        """

        dim: the length of the fixed-length vector

        """

        self.dim = dim

    def get_origin(self):

        return np.zeros((self.dim,))

    def get_origin_batch(self, n):

        return np.zeros((n,self.dim))

    def make_theano_batch(self, name = None, dtype = None):

        if dtype is None:
            dtype = config.floatX

        return T.matrix(name = name, dtype = dtype)


class Conv2DSpace(Space):
    """ Defines a space whose points are defined as multi-channel images """

    def __init__(self, shape, nchannels):
        """
            shape: (rows, cols)
            nchannels: # of channels in the image
        """

        self.shape = shape
        self.nchannels = nchannels

    def get_origin(self):
        return np.zeros((self.shape[0], self.shape[1], self.nchannels))

    def get_origin_batch(self, n):
        return np.zeros((n, self.shape[0], self.shape[1], self.nchannels))

    def make_theano_batch(self, name = None, dtype = None):

        if dtype is None:
            dtype = config.floatX

        return TensorType( dtype = dtype, broadcastable = (False, False, False, self.nchannels == 1) )(name = name)
