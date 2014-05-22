"""
This module is based on Alex Krizhevsky's cuda-convnet locally connected layers.
pylearn2.linear.conv2d is based on theano's 2D convolution.
This module therefore uses images with axis format ('c', 0, 1, 'b')
as its native format.
Unlike the other cuda-convnet functionality in pylearn2, this linear transform
has CPU support, provided by TheanoLinear.
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

import logging
import numpy as np
import warnings

from pylearn2.packaged_dependencies.theano_linear.unshared_conv.localdot import LocalDot

from pylearn2.utils import sharedX
from pylearn2.utils.rng import make_np_rng
from pylearn2.linear.conv2d import default_seed, default_sparse_seed
from pylearn2.linear.linear_transform import LinearTransform


logger = logging.getLogger(__name__)


class Local(LinearTransform, LocalDot):
    """
    A pylearn2 linear operator based on local receptive fields (convolution
    without parameter sharing) implemented using Alex Krizhevsky's cuda-convnet
    library + James Bergstra's TheanoLinear module

    Parameters
    ----------
    filters : WRITEME
    image_shape : WRITEME
    input_groups : WRITEME
    input_axes : WRITEME
    batch_size : WRITEME
    output_axes : WRITEME
    kernel_stride : WRITEME
    pad : WRITEME
    message : WRITEME
    partial_sum : WRITEME
    """

    def __init__(self, filters, image_shape, input_groups,
                 input_axes=('c', 0, 1, 'b'), batch_size=None,
                 output_axes=('c', 0, 1, 'b'), kernel_stride=(1, 1), pad=0,
                 message='', partial_sum=None):
        self.input_groups = input_groups

        """TODO: Local ignores partial_sum argument,
                 figure out how James' code controls it"""

        logger.warning("partial_sum argument ignored")

        LocalDot.__init__(self,
            filters=filters,
            irows=image_shape[0],
            icols=image_shape[1],
            subsample=kernel_stride,
            padding_start=pad,
            message='')


    def lmul(self, x):
        """
        .. todo::

            WRITEME
        """

        reshaped = x.reshape(( self.input_groups, x.shape[0] / self.input_groups, x.shape[1], x.shape[2], x.shape[3]))

        out = LocalDot.rmul(self, reshaped)

        return out.reshape((out.shape[0] * out.shape[1], out.shape[2], out.shape[3], out.shape[4]))

    def get_params(self):
        """
        .. todo::

            WRITEME
        """
        return [self._filters]


def make_random_local(irange, input_channels, input_axes, input_groups,
        image_shape,
        output_channels,
        output_axes,
        kernel_shape,
        kernel_stride = (1,1), pad=0, message = "", rng = None,
        partial_sum = None):
    """
    .. todo::

        WRITEME properly

    Creates a Local with random weights.
    """

    rng = make_np_rng(rng, default_seed, which_method='uniform')

    def num_pos(img, stride, kwidth):
        img = img + 2 * pad
        return (img - kwidth) // stride + 1

    num_row_pos = num_pos(image_shape[0], kernel_stride[0], kernel_shape[0])
    num_col_pos = num_pos(image_shape[1], kernel_stride[1], kernel_shape[1])

    assert input_channels % input_groups == 0
    colors_per_group = input_channels // input_groups
    assert output_channels % input_groups == 0
    filters_per_group = output_channels // input_groups

    W = sharedX( rng.uniform(-irange,irange,
        (num_row_pos, num_col_pos, colors_per_group, kernel_shape[0], kernel_shape[1], input_groups,
            filters_per_group)))

    return Local(filters = W,
        image_shape = image_shape,
        input_groups = input_groups,
        input_axes = input_axes,
        output_axes = output_axes,
        kernel_stride = kernel_stride, pad=pad,
        message = message, partial_sum=partial_sum)


def make_sparse_random_local(num_nonzero, input_space, output_space,
        kernel_shape, batch_size, \
        kernel_stride = (1,1), border_mode = 'valid', message = "", rng=None):
    """
    .. todo::

        WRITEME
    """
    raise NotImplementedError("Not yet modified after copy-paste from "
            "pylearn2.linear.conv2d_c01b")
    """ Creates a Conv2D with random kernels, where the randomly initialized
    values are sparse"""

    rng = make_np_rng(rng, default_sparse_seed, which_method=['randn','randint'])

    W = np.zeros(( output_space.num_channels, input_space.num_channels, \
            kernel_shape[0], kernel_shape[1]))

    def random_coord():
        return [ rng.randint(dim) for dim in W.shape ]

    for i in xrange(num_nonzero):
        o, ch, r, c = random_coord()
        while W[o, ch, r, c] != 0:
            o, ch, r, c = random_coord()
        W[o, ch, r, c] = rng.randn()


    W = sharedX( W)

    #return Conv2D(filters = W,
    #    batch_size = batch_size,
    #    input_space = input_space,
    #    output_axes = output_space.axes,
    #    kernel_stride = kernel_stride, border_mode = border_mode,
    #    filters_shape = W.get_value(borrow=True).shape, message = message)
