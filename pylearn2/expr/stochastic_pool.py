"""
An implementation of stochastic max-pooling, based on

Stochastic Pooling for Regularization of Deep Convolutional Neural Networks
Matthew D. Zeiler, Rob Fergus, ICLR 2013
"""

__authors__ = "Mehdi Mirza"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Mehdi Mirza", "Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Mehdi Mirza"
__email__ = "mirzamom@iro"

import numpy
import theano
from theano import tensor
from theano.gof.op import get_debug_values
from pylearn2.utils.rng import make_theano_rng
from pylearn2.utils import contains_inf

def stochastic_max_pool_bc01(bc01, pool_shape, pool_stride, image_shape, rng = None):
    """
    .. todo::

        WRITEME properly

    Stochastic max pooling for training as defined in:

    Stochastic Pooling for Regularization of Deep Convolutional Neural Networks
    Matthew D. Zeiler, Rob Fergus

    Parameters
    ----------
    bc01 : theano 4-tensor
        in format (batch size, channels, rows, cols),
        IMPORTANT: All values should be positive
    pool_shape : tuple
        shape of the pool region (rows, cols)
    pool_stride : tuple
        strides between pooling regions (row stride, col stride)
    image_shape : tuple
        avoid doing some of the arithmetic in theano
    rng : theano random stream
    """
    r, c = image_shape
    pr, pc = pool_shape
    rs, cs = pool_stride

    batch = bc01.shape[0]
    channel = bc01.shape[1]

    rng = make_theano_rng(rng, 2022, which_method='multinomial')

    # Compute index in pooled space of last needed pool
    # (needed = each input pixel must appear in at least one pool)
    def last_pool(im_shp, p_shp, p_strd):
        rval = int(numpy.ceil(float(im_shp - p_shp) / p_strd))
        assert p_strd * rval + p_shp >= im_shp
        assert p_strd * (rval - 1) + p_shp < im_shp
        return rval
    # Compute starting row of the last pool
    last_pool_r = last_pool(image_shape[0] ,pool_shape[0], pool_stride[0]) * pool_stride[0]
    # Compute number of rows needed in image for all indexes to work out
    required_r = last_pool_r + pr

    last_pool_c = last_pool(image_shape[1] ,pool_shape[1], pool_stride[1]) * pool_stride[1]
    required_c = last_pool_c + pc

    # final result shape
    res_r = int(numpy.floor(last_pool_r/rs)) + 1
    res_c = int(numpy.floor(last_pool_c/cs)) + 1

    for bc01v in get_debug_values(bc01):
        assert not contains_inf(bc01v)
        assert bc01v.shape[2] == image_shape[0]
        assert bc01v.shape[3] == image_shape[1]

    # padding
    padded = tensor.alloc(0.0, batch, channel, required_r, required_c)
    name = bc01.name
    if name is None:
        name = 'anon_bc01'
    bc01 = tensor.set_subtensor(padded[:,:, 0:r, 0:c], bc01)
    bc01.name = 'zero_padded_' + name

    # unraveling
    window = tensor.alloc(0.0, batch, channel, res_r, res_c, pr, pc)
    window.name = 'unravlled_winodows_' + name

    for row_within_pool in xrange(pool_shape[0]):
        row_stop = last_pool_r + row_within_pool + 1
        for col_within_pool in xrange(pool_shape[1]):
            col_stop = last_pool_c + col_within_pool + 1
            win_cell = bc01[:,:,row_within_pool:row_stop:rs, col_within_pool:col_stop:cs]
            window  =  tensor.set_subtensor(window[:,:,:,:, row_within_pool, col_within_pool], win_cell)

    # find the norm
    norm = window.sum(axis = [4, 5])
    norm = tensor.switch(tensor.eq(norm, 0.0), 1.0, norm)
    norm = window / norm.dimshuffle(0, 1, 2, 3, 'x', 'x')
    # get prob
    prob = rng.multinomial(pvals = norm.reshape((batch * channel * res_r * res_c, pr * pc)), dtype='float32')
    # select
    res = (window * prob.reshape((batch, channel, res_r, res_c,  pr, pc))).max(axis=5).max(axis=4)
    res.name = 'pooled_' + name

    return tensor.cast(res, theano.config.floatX)

def weighted_max_pool_bc01(bc01, pool_shape, pool_stride, image_shape, rng = None):
    """
    This implements test time probability weighted pooling defined in:

    Stochastic Pooling for Regularization of Deep Convolutional Neural Networks
    Matthew D. Zeiler, Rob Fergus

    Parameters
    ----------
    bc01 : theano 4-tensor
        minibatch in format (batch size, channels, rows, cols),
        IMPORTANT: All values should be poitivie
    pool_shape : theano 4-tensor
        shape of the pool region (rows, cols)
    pool_stride : tuple
        strides between pooling regions (row stride, col stride)
    image_shape : tuple
        avoid doing some of the arithmetic in theano
    """
    r, c = image_shape
    pr, pc = pool_shape
    rs, cs = pool_stride

    batch = bc01.shape[0]
    channel = bc01.shape[1]

    rng = make_theano_rng(rng, 2022, which_method='multinomial')

    # Compute index in pooled space of last needed pool
    # (needed = each input pixel must appear in at least one pool)
    def last_pool(im_shp, p_shp, p_strd):
        rval = int(numpy.ceil(float(im_shp - p_shp) / p_strd))
        assert p_strd * rval + p_shp >= im_shp
        assert p_strd * (rval - 1) + p_shp < im_shp
        return rval
    # Compute starting row of the last pool
    last_pool_r = last_pool(image_shape[0] ,pool_shape[0], pool_stride[0]) * pool_stride[0]
    # Compute number of rows needed in image for all indexes to work out
    required_r = last_pool_r + pr

    last_pool_c = last_pool(image_shape[1] ,pool_shape[1], pool_stride[1]) * pool_stride[1]
    required_c = last_pool_c + pc

    # final result shape
    res_r = int(numpy.floor(last_pool_r/rs)) + 1
    res_c = int(numpy.floor(last_pool_c/cs)) + 1

    for bc01v in get_debug_values(bc01):
        assert not contains_inf(bc01v)
        assert bc01v.shape[2] == image_shape[0]
        assert bc01v.shape[3] == image_shape[1]

    # padding
    padded = tensor.alloc(0.0, batch, channel, required_r, required_c)
    name = bc01.name
    if name is None:
        name = 'anon_bc01'
    bc01 = tensor.set_subtensor(padded[:,:, 0:r, 0:c], bc01)
    bc01.name = 'zero_padded_' + name

    # unraveling
    window = tensor.alloc(0.0, batch, channel, res_r, res_c, pr, pc)
    window.name = 'unravlled_winodows_' + name

    for row_within_pool in xrange(pool_shape[0]):
        row_stop = last_pool_r + row_within_pool + 1
        for col_within_pool in xrange(pool_shape[1]):
            col_stop = last_pool_c + col_within_pool + 1
            win_cell = bc01[:,:,row_within_pool:row_stop:rs, col_within_pool:col_stop:cs]
            window  =  tensor.set_subtensor(window[:,:,:,:, row_within_pool, col_within_pool], win_cell)

    # find the norm
    norm = window.sum(axis = [4, 5])
    norm = tensor.switch(tensor.eq(norm, 0.0), 1.0, norm)
    norm = window / norm.dimshuffle(0, 1, 2, 3, 'x', 'x')
    # average
    res = (window * norm).sum(axis=[4,5])
    res.name = 'pooled_' + name

    return res.reshape((batch, channel, res_r, res_c))
