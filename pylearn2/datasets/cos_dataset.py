"""
.. todo::

    WRITEME
"""
import numpy as N
import copy
from theano import config
import theano.tensor as T
from pylearn2.utils.rng import make_np_rng


class CosDataset(object):

    """
    Makes a dataset that streams randomly generated 2D examples.
    The first coordinate is sampled from a uniform distribution.
    The second coordinate is the cosine of the first coordinate,
    plus some gaussian noise.
    """

    def __init__(self, min_x=-6.28, max_x=6.28, std=.05, rng=None):
        """
        .. todo::

            WRITEME
        """
        self.min_x, self.max_x, self.std = min_x, max_x, std
        rng = make_np_rng(rng, [17, 2, 946], which_method=['uniform', 'randn'])
        self.default_rng = copy.copy(rng)
        self.rng = rng

    def energy(self, mat):
        """
        .. todo::

            WRITEME
        """
        x = mat[:, 0]
        y = mat[:, 1]
        rval = (y - N.cos(x)) ** 2. / (2. * (self.std ** 2.))
        return rval

    def pdf_func(self, mat):
        """
        .. todo::

            WRITEME properly

        This dataset can generate an infinite amount of examples.
        This function gives the pdf from which the examples are drawn.
        """
        x = mat[:, 0]
        y = mat[:, 1]
        rval = N.exp(-(y - N.cos(x)) ** 2. / (2. * (self.std ** 2.)))
        rval /= N.sqrt(2.0 * N.pi * (self.std ** 2.))
        rval /= (self.max_x - self.min_x)
        rval *= x < self.max_x
        rval *= x > self.min_x
        return rval

    def free_energy(self, X):
        """
        .. todo::

            WRITEME properly

        This dataset can generate an infinite amount of examples.
        This function gives the energy function for the distribution from
        which the examples are drawn.
        """
        x = X[:, 0]
        y = X[:, 1]
        rval = T.sqr(y - T.cos(x)) / (2. * (self.std ** 2.))
        mask = x < self.max_x
        mask = mask * (x > self.min_x)
        rval = mask * rval + (1 - mask) * 1e30
        return rval

    def pdf(self, X):
        """
        .. todo::

            WRITEME properly

        This dataset can generate an infinite amount of examples.
        This function gives the pdf from which the examples are drawn.
        """
        x = X[:, 0]
        y = X[:, 1]
        rval = T.exp(-T.sqr(y - T.cos(x)) / (2. * (self.std ** 2.)))
        rval /= N.sqrt(2.0 * N.pi * (self.std ** 2.))
        rval /= (self.max_x - self.min_x)
        rval *= x < self.max_x
        rval *= x > self.min_x
        return rval

    def get_stream_position(self):
        """
        .. todo::

            WRITEME
        """
        return copy.copy(self.rng)

    def set_stream_position(self, s):
        """
        .. todo::

            WRITEME
        """
        self.rng = copy.copy(s)

    def restart_stream(self):
        """
        .. todo::

            WRITEME
        """
        self.reset_RNG()

    def reset_RNG(self):
        """
        .. todo::

            WRITEME
        """
        if 'default_rng' not in dir(self):
            self.default_rng = N.random.RandomState([17, 2, 946])
        self.rng = copy.copy(self.default_rng)

    def apply_preprocessor(self, preprocessor, can_fit=False):
        """
        .. todo::

            WRITEME
        """
        raise NotImplementedError()

    def get_batch_design(self, batch_size):
        """
        .. todo::

            WRITEME
        """
        x = N.cast[config.floatX](self.rng.uniform(self.min_x, self.max_x,
                                                   (batch_size, 1)))
        y = N.cos(x) + (N.cast[config.floatX](self.rng.randn(*x.shape)) *
                        self.std)
        rval = N.hstack((x, y))
        return rval
