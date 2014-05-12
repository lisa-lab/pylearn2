"""
Code for normalizing outputs of MLP / convnet layers.
"""
__authors__ = "Ian Goodfellow and David Warde-Farley"
__copyright__ = "Copyright 2013, Universite de Montreal"
__credits__ = ["Ian Goodfellow and David Warde-Farley"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

import theano.tensor as T

from pylearn2.sandbox.cuda_convnet.response_norm import CrossMapNorm

class CrossChannelNormalizationBC01(object):
    """
    BC01 version of CrossChannelNormalization

    Parameters
    ----------
    alpha : WRITEME
    k : WRITEME
    beta : WRITEME
    n : WRITEME
    """

    def __init__(self, alpha = 1e-4, k=2, beta=0.75, n=5):
        self.__dict__.update(locals())
        del self.self

        if n % 2 == 0:
            raise NotImplementedError("Only works with odd n for now")

    def __call__(self, bc01):
        """
        .. todo::

            WRITEME
        """
        half = self.n // 2

        sq = T.sqr(bc01)

        b, ch, r, c = bc01.shape

        extra_channels = T.alloc(0., b, ch + 2*half, r, c)

        sq = T.set_subtensor(extra_channels[:,half:half+ch,:,:], sq)

        scale = self.k

        for i in xrange(self.n):
            scale += self.alpha * sq[:,i:i+ch,:,:]

        scale = scale ** self.beta

        return bc01 / scale

class CrossChannelNormalization(object):
    """
    See "ImageNet Classification with Deep Convolutional Neural Networks"
    Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton
    NIPS 2012

    Section 3.3, Local Response Normalization

    .. todo::

        WRITEME properly

    f(c01b)_[i,j,k,l] = c01b[i,j,k,l] / scale[i,j,k,l]

    scale[i,j,k,l] = (k + sqr(c01b)[clip(i-n/2):clip(i+n/2),j,k,l].sum())^beta

    clip(i) = T.clip(i, 0, c01b.shape[0]-1)

    Parameters
    ----------
    alpha : WRITEME
    k : WRITEME
    beta : WRITEME
    n : WRITEME
    """

    def __init__(self, alpha = 1e-4, k=2, beta=0.75, n=5):
        self.__dict__.update(locals())
        del self.self

        if n % 2 == 0:
            raise NotImplementedError("Only works with odd n for now")

    def __call__(self, c01b):
        """
        .. todo::

            WRITEME
        """
        half = self.n // 2

        sq = T.sqr(c01b)

        ch, r, c, b = c01b.shape

        extra_channels = T.alloc(0., ch + 2*half, r, c, b)

        sq = T.set_subtensor(extra_channels[half:half+ch,:,:,:], sq)

        scale = self.k

        for i in xrange(self.n):
            scale += self.alpha * sq[i:i+ch,:,:,:]

        scale = scale ** self.beta

        return c01b / scale

class CudaConvNetCrossChannelNormalization(object):
    """
    .. todo::

        WRITEME properly

    I kept the same parameter names where I was sure they
    actually are the same parameters (with respect to
    CrossChannelNormalization).

    Parameters
    ----------
    alpha : WRITEME
    beta : WRITEME
    size_f : WRITEME
    blocked : WRITEME
    """
    def __init__(self, alpha=1e-4, beta=0.75, size_f=5, blocked=True):
        self._op = CrossMapNorm(size_f=size_f, add_scale=alpha,
                                pow_scale=beta, blocked=blocked)

    def __call__(self, c01b):
        """
        .. todo::

            WRITEME properly

        NOTE: c01b must be CudaNdarrayType."""
        return self._op(c01b)[0]
