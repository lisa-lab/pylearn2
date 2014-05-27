"""
Local winner-take-all layer.
"""
__author__ = "Xia Da, Ian Goodfellow, Minh Ngoc Le"

from functools import wraps
import numpy
from theano import tensor as T
from pylearn2.models.mlp import Linear, Layer


def lwta(p, block_size):
    """
    Apply hard local winner-take-all on every rows of a theano matrix.

    Parameters
    ----------
    p: theano matrix
        Matrix on whose rows LWTA will be applied.
    block_size: int
        Number of units in each block.
    """
    batch_size = p.shape[0]
    num_filters = p.shape[1]
    num_blocks = num_filters // block_size
    w = p.reshape((batch_size, num_blocks, block_size))
    block_max = w.max(axis=2).dimshuffle(0, 1, 'x') * T.ones_like(w)
    max_mask = T.cast(w >= block_max, 'float32')
    indices = numpy.array(range(1, block_size+1))
    max_mask2 = max_mask * indices
    block_max2 = max_mask2.max(axis=2).dimshuffle(0, 1, 'x') * T.ones_like(w)
    max_mask3 = T.cast(max_mask2 >= block_max2, 'float32')
    w2 = w * max_mask3
    w3 = w2.reshape((p.shape[0], p.shape[1]))
    return w3


class LWTA(Linear):
    """
    An MLP Layer using the hard local winner-take-all non-linearity.

    The hard LWTA non-linearity is described in [1]. In short, only one unit
    in each block is activated, which is the one with maximal net input. Its
    activation is exactly its net input.

    .. [1] Srivastava, R. K., Masci, J., Kazerounian, S., Gomez, F., &
       Schmidhuber, J. (2013). Compete to compute. In Advances in Neural
       Information Processing Systems (pp. 2310-2318).
       URL: http://www.idsia.ch/idsiareport/IDSIA-04-13.pdf

    Parameters
    ----------
    block_size: int
        Number of units in each block.

    Notes
    ----------
    Our implementation differs slightly from Rupesh Srivastava et al.'s -- we
    break ties by last index, they break them by earliest index. This
    difference is just due to ease of implementation in theano.
    """

    def __init__(self, block_size, **kwargs):
        super(LWTA, self).__init__(**kwargs)
        self.block_size = block_size

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        p = super(LWTA, self).fprop(state_below)
        rval = lwta(p, self.block_size)
        rval.name = self.layer_name + '_out'
        return rval
