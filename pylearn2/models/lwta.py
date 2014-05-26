__author__ = "Xia Da, Ian Goodfellow, Minh Ngoc Le"
import numpy
from theano import tensor as T
from pylearn2.models.mlp import Linear

def lwta(p, block_size):
    """
    The hard local winner take all non-linearity from "Compete to Compute"
    by Rupesh Srivastava et al
    Our implementation differs slightly from theirs--we break ties by last index,
    they break them by earliest index. This difference is just due to ease
    of implementation in theano.
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
    An MLP Layer using the LWTA non-linearity.
    """
    def __init__(self, block_size, **kwargs):
        super(LWTA, self).__init__(**kwargs)
        self.block_size = block_size

    def fprop(self, state_below):
        p = super(LWTA, self).fprop(state_below)
        w = lwta(p, self.block_size)
        w.name = self.layer_name + '_out'
        return w
